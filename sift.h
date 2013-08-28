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
typedef struct ImageSt {        /*金字塔每一层*/
 
 float levelsigma;
 int levelsigmalength;
 float absolute_sigma;
 XAF_MAT **Level;       
} ImageLevels;

typedef struct ImageSt1 {      /*金字塔每一阶梯*/
 int row, col;          //Dimensions of image. 
 float subsample;
 ImageLevels *Octave;              
} ImageOctaves;


//keypoint数据结构，Lists of keypoints are linked by the "next" field.
typedef struct KeypointSt 
{
 float row, col; /* 反馈回原图像大小，特征点的位置 */
 float sx,sy;    /* 金字塔中特征点的位置*/
 int octave,level;/*金字塔中，特征点所在的阶梯、层次*/
 
 float scale, ori,mag; /*所在层的尺度sigma,主方向orientation (range [-PI,PI])，以及幅值*/
 float *descrip;       /*特征描述字指针：128维或32维等*/
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
XAF_MAT* halfSizeImage(XAF_MAT * im);     //缩小图像：下采样
XAF_MAT* doubleSizeImage(XAF_MAT * im);   //扩大图像：最近临方法
XAF_MAT* doubleSizeImage2(XAF_MAT * im);  //扩大图像：线性插值
float getPixelBI(XAF_MAT * im, float col, float row);//双线性插值函数
void normalizeVec(float* vec, int dim);//向量归一化  
XAF_MAT* GaussianKernel2D(float sigma);  //得到2维高斯核
void normalizeMat(XAF_MAT* mat) ;        //矩阵归一化
float* GaussianKernel1D(float sigma, int dim) ; //得到1维高斯核

//在具体像素处宽度方向进行高斯卷积
float ConvolveLocWidth(float* kernel, int dim, XAF_MAT * src, int x, int y) ;  
//在整个图像宽度方向进行1D高斯卷积
void Convolve1DWidth(float* kern, int dim, XAF_MAT * src, XAF_MAT * dst) ;       
//在具体像素处高度方向进行高斯卷积
float ConvolveLocHeight(float* kernel, int dim, XAF_MAT * src, int x, int y) ; 
//在整个图像高度方向进行1D高斯卷积
void Convolve1DHeight(float* kern, int dim, XAF_MAT * src, XAF_MAT * dst);     
//用高斯函数模糊图像  
int BlurImage(XAF_MAT * src, XAF_MAT * dst, float sigma) ;  

//SIFT算法第一步：图像预处理
XAF_MAT *ScaleInitImage(XAF_MAT * im) ;                  //金字塔初始化

//SIFT算法第二步：建立高斯金字塔函数
ImageOctaves* BuildGaussianOctaves(XAF_MAT * image) ;  //建立高斯金字塔

//SIFT算法第三步：特征点位置检测，最后确定特征点的位置
int DetectKeypoint(int numoctaves, ImageOctaves *GaussianPyr);
void DisplayKeypointLocation(int im, ImageOctaves *GaussianPyr);

//SIFT算法第四步：计算高斯图像的梯度方向和幅值，计算各个特征点的主方向
void ComputeGrad_DirecandMag(int numoctaves, ImageOctaves *GaussianPyr);

int FindClosestRotationBin (int binCount, float angle);  //进行方向直方图统计
void AverageWeakBins (double* bins, int binCount);       //对方向直方图滤波
//确定真正的主方向
bool InterpolateOrientation(double left, double middle,double right, double *degreeCorrection, double *peakValue);
//确定各个特征点处的主方向函数
void AssignTheMainOrientation(int numoctaves, ImageOctaves *GaussianPyr,ImageOctaves *mag_pyr,ImageOctaves *grad_pyr);
//显示主方向
void DisplayOrientation (int id, ImageOctaves *GaussianPyr);

//SIFT算法第五步：抽取各个特征点处的特征描述字
void ExtractFeatureDescriptors(int numoctaves, ImageOctaves *GaussianPyr);

//为了显示图象金字塔，而作的图像水平、垂直拼接
XAF_MAT* MosaicHorizen( XAF_MAT* im1, XAF_MAT* im2 );
XAF_MAT* MosaicVertical( XAF_MAT* im1, XAF_MAT* im2 );
int test(gdImagePtr im,int id TSRMLS_DC);
KeypointPtr* siftArray();
//特征描述点，网格  
#define GridSpacing 4
