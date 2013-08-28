#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "ext/gd/php_gd.h"
#include "ext/gd/libgd/gd.h"
#include "php_xaf.h"
#define NUMSIZE 2
#define GAUSSKERN 3.5
#define PI 3.14159265358979323846
#define XAF_32FC1 5
#define XAF_8FC1 1
#define XAF_64FC1 8
#define XAF_16FC1 2
#define XAF_32FC 4
#define COPYIMAGE(ids,idd,src,dst) copyImage(ids,idd,0,0,0,0,gdImageSX(dst), gdImageSY(dst),gdImageSX(src), gdImageSY(src) TSRMLS_CC);
#define CREATEIMAGE(x,y,dst)  createImage_ex(x,y,dst TSRMLS_CC)
#ifndef bool
#define bool int
#endif
#define true 1
#define false 0
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE 
#define FALSE 0
#endif
#ifndef min
#define min(a,b) ((a)<(b) ? (a):(b))
#define max(a,b) ((a)>(b) ? (a):(b))
#endif
//Sigma of base image -- See D.L.'s paper.
#define INITSIGMA 0.5
//Sigma of each octave -- See D.L.'s paper.
#define SIGMA sqrt(3)//1.6//

//Number of scales per octave.  See D.L.'s paper.
#define SCALESPEROCTAVE 2
#define MAXOCTAVES 4

#define CONTRAST_THRESHOLD   3000.0
#define CURVATURE_THRESHOLD  10.0
#define DOUBLE_BASE_IMAGE_SIZE 1
#define peakRelThresh 0.8
#define LEN 128
#ifndef NULL
#define NULL 0
#endif
#define XAF_MAT xaf_mat
#define Keypoint KeypointPtr*
#define XAF_ARGB(r,g,b,a) ((a<<24)|(r<<16)|(g<<8)|(b))
#define XAF_RGB(r,g,b) ((0<<24)|(r<<16)|(g<<8)|(b))
#define XAF_1D(i,sigma) ((float)(1.0f/(sqrt(2.0f * PI) * sigma))* exp(-(1.0*i*i) / (2.0 * sigma*sigma))) 
#define XAF_2D(i,j,sigma) ((float)(1.0f/(sqrt(2.0f * PI) * sigma))* exp(-(1.0*i*i + 1.0*j*j) / (2.0 * sigma*sigma)))
#define SIMGA2  1.4142135623731
#define SIMGAP 2.0
// temporary storage
//CvMemStorage* storage = 0;

typedef struct _xaf_mat
{
	int type;
	int cols;
	int rows;
	int step;
	union{
		unsigned char** ptr;
		short** s;
		int** i;
		float** fl;
		double** db;
	} data;
} xaf_mat;

//Data structure for a float image.
typedef struct ImageSt {        /*������ÿһ��*/
 
 float levelsigma;
 int levelsigmalength;
 float absolute_sigma;
 XAF_MAT **Level;       
} ImageLevels;

typedef struct ImageSt1 {      /*������ÿһ����*/
 int row, col;          //Dimensions of image. 
 float subsample;
 ImageLevels *Octave;              
} ImageOctaves;


//keypoint���ݽṹ��Lists of keypoints are linked by the "next" field.
typedef struct KeypointSt 
{
 float row, col; /* ������ԭͼ���С���������λ�� */
 float sx,sy;    /* ���������������λ��*/
 int octave,level;/*�������У����������ڵĽ��ݡ����*/
 
 float scale, ori,mag; /*���ڲ�ĳ߶�sigma,������orientation (range [-PI,PI])���Լ���ֵ*/
 float *descrip;       /*����������ָ�룺128ά��32ά��*/
 struct KeypointSt *next;/* Pointer to next keypoint in list. */
} KeypointPtr;


extern ImageOctaves *DOGoctaves; 
XAF_MAT* createMat(int h,int w,int type);
zval* createImage(int x,int y TSRMLS_DC );
gdImagePtr createImage_ex(int x,int y,int* dst TSRMLS_DC );
void xafConvert(gdImagePtr im,XAF_MAT* mat);
bool xafConvertScale(XAF_MAT* src,XAF_MAT *dst,float scale,int shift);
void releaseMat(XAF_MAT **mat);
void xafConvert2Image(XAF_MAT* mat,gdImagePtr im);
XAF_MAT* halfSizeImage(XAF_MAT * im);     //��Сͼ���²���
XAF_MAT* doubleSizeImage(XAF_MAT * im);   //����ͼ������ٷ���
XAF_MAT* doubleSizeImage2(XAF_MAT * im);  //����ͼ�����Բ�ֵ
float getPixelBI(XAF_MAT * im, float col, float row);//˫���Բ�ֵ����
void normalizeVec(float* vec, int dim);//������һ��  
XAF_MAT* GaussianKernel2D(float sigma);  //�õ�2ά��˹��
void normalizeMat(XAF_MAT* mat) ;        //�����һ��
float* GaussianKernel1D(float sigma, int dim) ; //�õ�1ά��˹��

//�ھ������ش���ȷ�����и�˹���
float ConvolveLocWidth(float* kernel, int dim, XAF_MAT * src, int x, int y) ;  
//������ͼ���ȷ������1D��˹���
void Convolve1DWidth(float* kern, int dim, XAF_MAT * src, XAF_MAT * dst) ;       
//�ھ������ش��߶ȷ�����и�˹���
float ConvolveLocHeight(float* kernel, int dim, XAF_MAT * src, int x, int y) ; 
//������ͼ��߶ȷ������1D��˹���
void Convolve1DHeight(float* kern, int dim, XAF_MAT * src, XAF_MAT * dst);     
//�ø�˹����ģ��ͼ��  
int BlurImage(XAF_MAT * src, XAF_MAT * dst, float sigma) ;  

//SIFT�㷨��һ����ͼ��Ԥ����
XAF_MAT *ScaleInitImage(XAF_MAT * im) ;                  //��������ʼ��

//SIFT�㷨�ڶ�����������˹����������
ImageOctaves* BuildGaussianOctaves(XAF_MAT * image) ;  //������˹������

//SIFT�㷨��������������λ�ü�⣬���ȷ���������λ��
int DetectKeypoint(int numoctaves, ImageOctaves *GaussianPyr);
void DisplayKeypointLocation(int im, ImageOctaves *GaussianPyr);

//SIFT�㷨���Ĳ��������˹ͼ����ݶȷ���ͷ�ֵ����������������������
void ComputeGrad_DirecandMag(int numoctaves, ImageOctaves *GaussianPyr);

int FindClosestRotationBin (int binCount, float angle);  //���з���ֱ��ͼͳ��
void AverageWeakBins (double* bins, int binCount);       //�Է���ֱ��ͼ�˲�
//ȷ��������������
bool InterpolateOrientation(double left, double middle,double right, double *degreeCorrection, double *peakValue);
//ȷ�����������㴦����������
void AssignTheMainOrientation(int numoctaves, ImageOctaves *GaussianPyr,ImageOctaves *mag_pyr,ImageOctaves *grad_pyr);
//��ʾ������
void DisplayOrientation (int id, ImageOctaves *GaussianPyr);

//SIFT�㷨���岽����ȡ���������㴦������������
void ExtractFeatureDescriptors(int numoctaves, ImageOctaves *GaussianPyr);

//Ϊ����ʾͼ���������������ͼ��ˮƽ����ֱƴ��
XAF_MAT* MosaicHorizen( XAF_MAT* im1, XAF_MAT* im2 );
XAF_MAT* MosaicVertical( XAF_MAT* im1, XAF_MAT* im2 );
int test(gdImagePtr im,int id TSRMLS_DC);
KeypointPtr* siftArray();
//���������㣬����  
#define GridSpacing 4
