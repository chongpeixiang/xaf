#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "Zend/zend_interfaces.h"
#include "sift.h"
#define IMAT(IM,ROW,COL) (IM->data.fl[(ROW)][(COL)]) 
#define ImLevels( DOGoctaves,OCTAVE,LEVEL,ROW,COL) IMAT((DOGoctaves)[(OCTAVE)].Octave[(LEVEL)].Level[0],ROW,COL)

ImageOctaves *DOGoctaves = NULL;      
//DOG pyr，DOG算子计算简单，是尺度归一化的LoG算子的近似。
ImageOctaves *mag_thresh ;
ImageOctaves *mag_pyr ;
ImageOctaves *grad_pyr ;
int     numoctaves;
//定义特征点具体变量
Keypoint keypoints=NULL;      //用于临时存储特征点的位置等
Keypoint keyDescriptors=NULL; //用于最后的确定特征点以及特征描述字
xaf_mat* createMat(int h,int w,int type){
	xaf_mat *mat;
	mat = (xaf_mat*)malloc(sizeof(xaf_mat));
	mat->type = type;
	mat->cols = w;
	mat->rows = h;
	if(!type){
		type = 5;
	}
	switch(type)
	{
	case XAF_8FC1: mat->data.ptr = (unsigned char**)malloc(sizeof(char*)*h);VALARR_INIT(mat->data.ptr,w,h,unsigned char); mat->step = sizeof(unsigned char)*w;break;
		case XAF_16FC1: mat->data.s = (short**)malloc(sizeof(short*)*h);VALARR_INIT(mat->data.s,w,h,short); mat->step = sizeof(short)*w;break;
		case XAF_32FC: mat->data.i = (int**)malloc(sizeof(int*)*h);VALARR_INIT(mat->data.i,w,h,int); mat->step = sizeof(int)*w;break;
		case XAF_32FC1: mat->data.fl = (float**)malloc(sizeof(float*)*h);VALARR_INIT(mat->data.fl,w,h,float);mat->step = sizeof(float)*w; break;
		case XAF_64FC1: mat->data.db = (double**)malloc(sizeof(double*)*h);VALARR_INIT(mat->data.db,w,h,double);mat->step = sizeof(double)*w; break;
	}

	return mat;
}

XAF_MAT *cloneMat(XAF_MAT* im){
	int i;
	XAF_MAT *newim;
	newim = createMat(im->rows,im->cols,im->type);
	switch(im->type){

	case XAF_8FC1:for(i=0;i<im->rows;i++){
		memcpy(newim->data.ptr[i],im->data.ptr[i],sizeof(unsigned char)*im->cols);
	} 
	break;
	case XAF_16FC1:for(i=0;i<im->rows;i++){
		memcpy(newim->data.s[i],im->data.s[i],sizeof(short)*im->cols);
	} 
	break;
	case XAF_32FC:for(i=0;i<im->rows;i++){
		memcpy(newim->data.i[i],im->data.i[i],sizeof(int)*im->cols);
	} 
	break;
	case XAF_32FC1:for(i=0;i<im->rows;i++){
		memcpy(newim->data.fl[i],im->data.fl[i],sizeof(float)*im->cols);
	} 
	break;
	case XAF_64FC1:for(i=0;i<im->rows;i++){
		memcpy(newim->data.db[i],im->data.db[i],sizeof(double)*im->cols);
	} 
	break;

	}
	return newim;
}

static void xafSub(XAF_MAT* src1,XAF_MAT* src2,XAF_MAT* dst,XAF_MAT* mask){
	int i,j;
	XAF_MAT *tempMat;
	if((src1->rows != src2->rows) || (src1->cols != src2->cols)){
		zend_error(E_ERROR,"mem length small");
	}
	if(!mask){
		for(i = 0 ;i< dst->rows;i++){
			for(j = 0; j< dst->cols;j++){
				IMAT(dst,i,j) = IMAT(src1,i,j) - IMAT(src2,i,j);
			}
		}
	}

}

gdPoint* xafPoint(int x,int y)
{
	gdPoint *p = (gdPoint*)malloc(sizeof(gdPoint));
	p->x = x;
	p->y = y;
	return p;
}
void xafZero(xaf_mat* im){
	int i ;
	switch(im->type){

	case XAF_8FC1:for(i=0;i<im->rows;i++){
		memset(im->data.ptr[i],0,sizeof(unsigned char)*im->cols);
	} 
	break;
	case XAF_16FC1:for(i=0;i<im->rows;i++){
		memset(im->data.s[i],0,sizeof(short)*im->cols);
	} 
	break;
	case XAF_32FC:for(i=0;i<im->rows;i++){
		memset(im->data.i[i],0,sizeof(int)*im->cols);
	} 
	break;
	case XAF_32FC1:for(i=0;i<im->rows;i++){
		memset(im->data.fl[i],0,sizeof(float)*im->cols);
	} 
	break;
	case XAF_64FC1:for(i=0;i<im->rows;i++){
		memset(im->data.db[i],0,sizeof(double)*im->cols);
	} 
	break;

	}

}

void xafLine(int im,gdPoint* p1,gdPoint* p2,int color,int thickness,int line_type,int shift TSRMLS_DC ){
	zval func,*ret_zval,*param[6],***params_array;
	int i;
	ZVAL_STRING(&func, "imageline", 1);

	ZVALARR_INIT(param,6);
	ZVAL_RESOURCE(param[0],im);
	ZVAL_LONG(param[1],p1->x);
	ZVAL_LONG(param[2],p1->y);
	ZVAL_LONG(param[3],p2->x);
	ZVAL_LONG(param[4],p2->y);
	ZVAL_LONG(param[5],color);
	params_array = (zval ***) emalloc(sizeof(zval **)*10);
		for (i=0; i<6; i++) {
			params_array[i] = &param[i];
	}

	call_user_function_ex(CG(function_table), NULL, &func, &ret_zval, 6, params_array,1,NULL TSRMLS_CC);
	ZVALARR_FREE(param,6);
	efree(params_array);
/*	zendi_zval_dtor(*param[0]);*/
}

zval* createImage(int x,int y TSRMLS_DC){
	gdImagePtr im;
	zval func,*ret_zval=NULL,*param[2],*res,*return_value,***params_array;
	int i;
	ZVAL_STRING(&func, "imagecreatetruecolor", 1);//imagecreatetruecolor
	
	MAKE_STD_ZVAL(param[0]);
	MAKE_STD_ZVAL(param[1]);

	ZVAL_LONG(param[0],x);
	ZVAL_LONG(param[1],y);
	params_array = (zval ***) emalloc(sizeof(zval **)*2);
		for (i=0; i<2; i++) {
			params_array[i] = &param[i];
	}

	call_user_function_ex(CG(function_table), NULL, &func, &ret_zval, 2, params_array,1,NULL TSRMLS_CC);
	ZVALARR_FREE(param,2);
	efree(params_array);
	//ZEND_FETCH_RESOURCE(im, gdImagePtr, &ret_zval, -1, "Image",phpi_get_le_gd());
	return ret_zval;
}

gdImagePtr createImage_ex(int x,int y,int* dst TSRMLS_DC){
	zval *rs,*return_value;
	gdImagePtr im;
	rs = createImage(x,y TSRMLS_CC);
	ZEND_FETCH_RESOURCE(im, gdImagePtr, &rs, -1, "Image",phpi_get_le_gd());
	*dst = Z_RESVAL_P(rs);
	return im;
}

void xafimagedestroy(int id  TSRMLS_DC)
{
	zval func,*ret_zval,*param[1],*res,*return_value,***params_array;
	int i;
	ZVAL_STRING(&func, "imagedestroy", 1);
	
	ZVALARR_INIT(param,1);
	ZVAL_RESOURCE(param[0],id);
	
	params_array = (zval ***) emalloc(sizeof(zval **)*1);
			params_array[0] = &param[0];

	call_user_function_ex(CG(function_table), NULL, &func, &ret_zval, 1, params_array,1,NULL TSRMLS_CC);
	ZVALARR_FREE(param,1);
	efree(params_array);
}

void copyImage(int src,int dst,int dx,int dy,int sx,int sy,int dw,int dh,int sw,int sh TSRMLS_DC ){
	gdImagePtr im;
	int id = phpi_get_le_gd();
	zval func,*ret_zval,*param[10],*res,*return_value,***params_array;
	int i;
	ZVAL_STRING(&func, "imagecopyresampled", 1);
	
	ZVALARR_INIT(param,10);
	ZVAL_RESOURCE(param[0],src);
	ZVAL_RESOURCE(param[1],dst);
	ZVAL_LONG(param[2],dx);
	ZVAL_LONG(param[3],dy);
	ZVAL_LONG(param[4],sx);
	ZVAL_LONG(param[5],sy);
	ZVAL_LONG(param[6],dw);
	ZVAL_LONG(param[7],dh);
	ZVAL_LONG(param[8],sw);
	ZVAL_LONG(param[9],sh);
	params_array = (zval ***) emalloc(sizeof(zval **)*10);
		for (i=0; i<10; i++) {
			params_array[i] = &param[i];
	}

	call_user_function_ex(CG(function_table), NULL, &func, &ret_zval, 10, params_array,1,NULL TSRMLS_CC);
	ZVALARR_FREE(param,10);
	efree(params_array);
	
}

void xafConvert(gdImagePtr im,XAF_MAT* mat)
{
	int i,j;
	for(j=0;j<mat->rows;j++){
		for(i=0;i<mat->cols;i++){
			
			IMAT(mat,j,i) = (float)colorPoint(im,i,j);
		}
	}
}
bool xafConvertScale(XAF_MAT* src,XAF_MAT *dst,float scale,int shift)
{
	int i,j;
	if(!scale){
		scale = 1;
	}
	if(!shift){
		shift = 0;
	}
	for(j=0;j<dst->rows;j++){
		for(i=0;i<dst->cols;i++)
			IMAT(dst,j,i) = IMAT(src,j,i)*scale + shift;
	}

	return TRUE;
	
}
void releaseMat(XAF_MAT **mat)
{
	switch((*mat)->type)
	{
		case XAF_8FC1: VALARR_FREE((*mat)->data.ptr,(*mat)->rows);free((*mat)->data.ptr);break;
		case XAF_16FC1: VALARR_FREE((*mat)->data.s,(*mat)->rows);free((*mat)->data.s);break;
		case XAF_32FC: VALARR_FREE((*mat)->data.i,(*mat)->rows);free((*mat)->data.i);break;
		case XAF_32FC1: VALARR_FREE((*mat)->data.fl,(*mat)->rows);free((*mat)->data.fl);break;
		case XAF_64FC1: VALARR_FREE((*mat)->data.db,(*mat)->rows);free((*mat)->data.db);break;
	}
	
}
void xafConvert2Image(XAF_MAT* mat,gdImagePtr im){
	int i,j;
	for(j=0;j<gdImageSY(im);j++){
		for(i=0;i<gdImageSX(im);i++){

			 colorChange(im,IMAT(mat,j,i),i,j);
		}
	}
}
//下采样原来的图像，返回缩小2倍尺寸的图像 
XAF_MAT * halfSizeImage(XAF_MAT * im)   
{  
 int i,j;  
 int w = im->cols/2;  
 int h = im->rows/2;   
 XAF_MAT *imnew = createMat(h, w,XAF_32FC1);  
 for ( j = 0; j < h; j++)   
  for ( i = 0; i < w; i++)   
   IMAT(imnew,j,i)=IMAT(im,j*2, i*2);  
  return imnew;  
}  
  
//上采样原来的图像，返回放大2倍尺寸的图像  
XAF_MAT * doubleSizeImage(XAF_MAT * im)   
{  
 int i,j;  
 int w = im->cols*2;  
 int h = im->rows*2;   
 XAF_MAT *imnew = createMat(h, w, XAF_32FC1);  
     
 for ( j = 0; j < h; j++){   
  for ( i = 0; i < w; i++){
   IMAT(imnew,j,i)=IMAT(im,j/2, i/2);  
  }
 }
  return imnew;  
}  
  
//上采样原来的图像，返回放大2倍尺寸的线性插值图像  
XAF_MAT * doubleSizeImage2(XAF_MAT * im)   
{  
 int i,j;  
 int w = im->cols*2;  
 int h = im->rows*2;   
 XAF_MAT *imnew = createMat(h, w, XAF_32FC1);  
 // fill every pixel so we don't have to worry about skipping pixels later  
 for ( j = 0; j < h; j++)   
 {  
  for ( i = 0; i < w; i++)   
  { 
   IMAT(imnew,j,i)=IMAT(im,j/2, i/2);
  }  
 }  
 /* 
 A B C 
 E F G 
 H I J 
 pixels A C H J are pixels from original image 
 pixels B E G I F are interpolated pixels 
 */ 
 // interpolate pixels B and I  
 for ( j = 0; j < h; j += 2)  
  for ( i = 1; i < w - 1; i += 2)  
   IMAT(imnew,j,i)=(float)0.5*(IMAT(im,j/2, i/2)+IMAT(im,j/2, i/2+1));  
  // interpolate pixels E and G  
  for ( j = 1; j < h - 1; j += 2)  
   for ( i = 0; i < w; i += 2)  
    IMAT(imnew,j,i)=(float)0.5*(IMAT(im,j/2, i/2)+IMAT(im,j/2+1, i/2));  
   // interpolate pixel F  
   for ( j = 1; j < h - 1; j += 2)  
    for ( i = 1; i < w - 1; i += 2)  
     IMAT(imnew,j,i)=0.25f*(IMAT(im,j/2, i/2)+IMAT(im,j/2+1, i/2)+IMAT(im,j/2, i/2+1)+IMAT(im,j/2+1, i/2+1));  
    return imnew;  
}  
  
//双线性插值，返回像素间的灰度值  
float getPixelBI(XAF_MAT * im, float col, float row)   
{  
 int irow, icol;  
 float rfrac, cfrac;  
 float row1 = 0, row2 = 0;  
 int width=im->cols;  
 int height=im->rows;  
 irow = (int) row;  
 icol = (int) col;  
   
 if (irow < 0 || irow >= height || icol < 0 || icol >= width)  
	return 0;  
 if (row > height - 1)  
	row = (float)(height - 1);  
 if (col > width - 1)  
	col = (float)(width - 1);  
 rfrac = 1.0f - (row - (float) irow);  
 cfrac = 1.0f - (col - (float) icol);  
 if (cfrac < 1)   
 {  
  row1 = cfrac * IMAT(im,irow,icol) + (1.0f - cfrac) * IMAT(im,irow,icol+1);  
 }   
 else   
 {  
  row1 = IMAT(im,irow,icol);  
 }  
 if (rfrac < 1)   
 {  
  if (cfrac < 1)   
  {  
   row2 = cfrac * IMAT(im,irow+1,icol) + (1.0f - cfrac) * IMAT(im,irow+1,icol+1);  
  } else   
  {  
   row2 = IMAT(im,irow+1,icol);  
  }  
 }  
 return rfrac * row1 + (1.0f - rfrac) * row2;  
}  
  
//矩阵归一化  
void normalizeMat(XAF_MAT* mat)   
{  
//#define Mat(ROW,COL) (mat->data.fl[(ROW)][(COL)]) 
 float sum = 0;  
 int i,j;  
 for (j = 0; j < mat->rows; j++)   
  for (i = 0; i < mat->cols; i++)   
   sum += IMAT(mat,j,i);  
  for ( j = 0; j < mat->rows; j++)   
   for (i = 0; i < mat->rows; i++)   
    IMAT(mat,j,i) /= sum;  
}  
  
//得到向量的欧式长度，2-范数  
float GetVecNorm( float* vec, int dim )  
{
 int i;
 float sum=0.0;  
 for ( i=0;i<dim;i++)  
  sum+=vec[i]*vec[i];  
    return (float)sqrt(sum);  
}  
  
//产生1D高斯核  
float* GaussianKernel1D(float sigma, int dim)   
{  
   
	int i;
	const int SMALL_GAUSSIAN_SIZE = 7;
	static const float small_gaussian_tab[][7] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };
	double x,t;
    double sigmaX = sigma > 0 ? sigma : ((dim-1)*0.5 - 1)*0.3 + 0.8;//当sigma小于0时，采用公式得到sigma(只与n有关)
    double scale2X = -0.5/(sigmaX*sigmaX);//高斯表达式后面要用到
    double sum = 0;
/*if  sigma < 0，且n为不大于7的奇整数，则核的滤波系数固定了，其固定在数组
small_gaussian_tab中，根据其n的长度来选择具体的值 ，如果不满足上面的，则固定核为0
        固定核为0表示自己计算其核*/ 
        
    const float* fixed_kernel = dim % 2 == 1 && dim <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
        small_gaussian_tab[dim>>1] : 0;

 //printf("GaussianKernel1D(): Creating 1x%d vector for sigma=%.3f gaussian kernel/n", dim, sigma);  
	float *kern=(float*)malloc( dim*sizeof(float) );  
 
	 for ( i = 0; i < dim; i++)   
	 {  
		 x = i - (dim-1)*0.5;
			//如果自己算其核的话，就常用公式exp(scale2X*x*x)计算，否则就用固定系数的核
		 t = fixed_kernel ? (double)fixed_kernel[i] : exp(scale2X*x*x);
       
				kern[i] = (float)t;//单精度要求时存入cf数组中
				sum += kern[i];//进行归一化时要用到
	 }  
	 sum = 1.0/sum;//归一化时核中各元素之和为1
    for( i = 0; i < dim; i++ )
    {
        
            kern[i] = (float)(kern[i]*sum);//归一化后的单精度核元素
			
    }

	return kern;  
}  
  
//产生2D高斯核矩阵  
XAF_MAT* GaussianKernel2D(float sigma)   
{  
 // int dim = (int) max(3.0f, GAUSSKERN * sigma);  
    int dim = (int) max(3.0f, 2.0 * GAUSSKERN *sigma + 1.0f);
	int c = dim / 2,i,j;  
	float v;
	XAF_MAT *mat;
 // make dim odd  
	 if (dim % 2 == 0)  
		dim++;  
 //printf("GaussianKernel(): Creating %dx%d matrix for sigma=%.3f gaussian/n", dim, dim, sigma);  
	mat=createMat(dim, dim, XAF_32FC1);   

	 for (i = 0; i < (dim + 1) >>1; i++)   
	 {  
	  for (j = 0; j < (dim + 1)>>1; j++)   
	  {  
	   //printf("%d %d %d/n", c, i, j);  
	   v = XAF_2D(i,j,sigma);  
	   IMAT(mat,c+i,c+j) =v;  
	   IMAT(mat,c-i,c+j) =v;  
	   IMAT(mat,c+i,c-j) =v;  
	   IMAT(mat,c-i,c-j) =v;  
	  }  

	}
 // normalizeMat(mat);  
	return mat;  
}  
  
//x方向像素处作卷积  
float ConvolveLocWidth(float* kernel, int dim, XAF_MAT * src, int x, int y)   
{  

	int i;  
	float pixel = 0;  
    int col;  
	int cen = dim>>1;  

 for ( i = 0; i < dim; i++)   
 {  
  col = x + (i - cen);  
  if (col < 0)  
   col = 0;  
  if (col >= src->cols)  
   col = src->cols - 1;  
  pixel += kernel[i] * IMAT(src,y,col);  
 }  

 return pixel;  
}  
  
//x方向作卷积  
void Convolve1DWidth(float* kern, int dim, XAF_MAT * src, XAF_MAT * dst)   
{  

  int i,j;  
   
 for ( j = 0; j < src->rows; j++)   
 {  
  for ( i = 0; i < src->cols; i++)   
  {  
	  IMAT(dst,j,i) = ConvolveLocWidth(kern, dim, src, i, j);  
  }  
 }  
}  
  
//y方向像素处作卷积  
float ConvolveLocHeight(float* kernel, int dim, XAF_MAT * src, int x, int y)   
{  

	int j;  
	float pixel = 0;  
	int cen = dim>>1,row;  
 
	 for ( j = 0; j < dim; j++)   
	 {  
	  row = y + (j - cen);  
	  if (row < 0)  
		row = 0;  
	  if (row >= src->rows)  
		row = src->rows - 1;  
	  pixel += kernel[j] * IMAT(src,row,x);  
	 }  
 return pixel;  
}  
  
//y方向作卷积  
void Convolve1DHeight(float* kern, int dim, XAF_MAT * src, XAF_MAT * dst)   
{  

 
  int i,j;  
 for ( j = 0; j < src->rows; j++)   
 {  
  for ( i = 0; i < src->cols; i++)   
  {  
   //printf("%d, %d/n", i, j);  
   IMAT(dst,j,i) = ConvolveLocHeight(kern, dim, src, i, j);  
  }  
 }  
}  
  
//卷积模糊图像  
int BlurImage(XAF_MAT * src, XAF_MAT * dst, float sigma)   
{  
 float* convkernel;  
 int dim = (int) max(3.0f, 2.0 * GAUSSKERN * sigma + 1.0f);  
 XAF_MAT *tempMat;  
 // make dim odd  
 if ((dim - dim>>1<<1) == 0)  
	dim++;  
 tempMat = createMat(src->rows, src->cols, XAF_32FC1); 
 convkernel = GaussianKernel1D(sigma, dim);  
   
 Convolve1DWidth(convkernel, dim, src,tempMat);  
 Convolve1DHeight(convkernel, dim, tempMat, dst); 

 releaseMat(&tempMat);  
 return dim;  
} 

XAF_MAT *ScaleInitImage(XAF_MAT * im)   
{  
    double sigma,preblur_sigma;  
	 XAF_MAT *imMat;  
	 XAF_MAT * dst;  
	 XAF_MAT *tempMat;  
 //首先对图像进行平滑滤波，抑制噪声  
	imMat = createMat(im->rows, im->cols, XAF_32FC1);  
   BlurImage(im, imMat, INITSIGMA);  
 //针对两种情况分别进行处理：初始化放大原始图像或者在原图像基础上进行后续操作  
 //建立金字塔的最底层  
 if (DOUBLE_BASE_IMAGE_SIZE)   
 {  
  tempMat = doubleSizeImage2(imMat);//对扩大两倍的图像进行二次采样，采样率为0.5，采用线性插值  
  dst = createMat(tempMat->rows, tempMat->cols, XAF_32FC1);  
  preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
  BlurImage(tempMat, dst, (float)preblur_sigma);   
    
  // The initial blurring for the first image of the first octave of the pyramid.  
  sigma = sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );   
  BlurImage(dst, tempMat,(float) sigma);       //得到金字塔的最底层-放大2倍的图像  
  releaseMat(& dst );   
  return tempMat;  
 }   
 else   
 {  
  dst = createMat(im->rows, im->cols, XAF_32FC1);  
  //sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA);  
  preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
  sigma = sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
  //printf("Init Sigma: %f/n", sigma);  
  BlurImage(imMat, dst,(float) sigma);        //得到金字塔的最底层：原始图像大小  
  return dst;  
 }   
}  

//SIFT算法第二步  
ImageOctaves* BuildGaussianOctaves(XAF_MAT * image)   
{  
    ImageOctaves *octaves;  
	XAF_MAT *tempMat;  
    XAF_MAT *dst;  
	XAF_MAT *temp;  
   
	int i,j;  
	double k = SIMGA2;  //方差倍数  
	float preblur_sigma, initial_sigma , sigma1,sigma2,sigma,absolute_sigma,sigma_f;  
 //计算金字塔的阶梯数目  
	int dim = min(image->rows, image->cols);  
	int numoctaves = (int) (log((double) dim) / log(2.0)) - 2;    //金字塔阶数  
	int length;
 //限定金字塔的阶梯数  
 numoctaves = min(numoctaves, MAXOCTAVES);  
 //为高斯金塔和DOG金字塔分配内存  
 octaves=(ImageOctaves*) malloc( numoctaves * sizeof(ImageOctaves) );  
 DOGoctaves=(ImageOctaves*) emalloc( numoctaves * sizeof(ImageOctaves) );  
    // start with initial source image  
 tempMat=cloneMat( image );
 
 // preblur_sigma = 1.0;//sqrt(2 - 4*INITSIGMA*INITSIGMA);  
 initial_sigma = SIMGA2;//sqrt( (4*INITSIGMA*INITSIGMA) + preblur_sigma * preblur_sigma );  
 //   initial_sigma = sqrt(SIGMA * SIGMA - INITSIGMA * INITSIGMA * 4);  
      
 //在每一阶金字塔图像中建立不同的尺度图像  
 for ( i = 0; i < numoctaves; i++)   
 {     
  //首先建立金字塔每一阶梯的最底层，其中0阶梯的最底层已经建立好  
//为各个阶梯分配内存  
  octaves[i].Octave= (ImageLevels*) malloc( (SCALESPEROCTAVE + 3) * sizeof(ImageLevels) );  
  DOGoctaves[i].Octave= (ImageLevels*) malloc( (SCALESPEROCTAVE + 2) * sizeof(ImageLevels) );  
  //存储各个阶梯的最底层  
  (octaves[i].Octave)[0].Level = &tempMat;     
  octaves[i].col=tempMat->cols;  
  octaves[i].row=tempMat->rows;  
  DOGoctaves[i].col=tempMat->cols;  
  DOGoctaves[i].row=tempMat->rows;  
  if (DOUBLE_BASE_IMAGE_SIZE)  
	  octaves[i].subsample=(float)pow((float)2,i)*0.5f;  
  else  
      octaves[i].subsample=(float)pow((float)2,i);  
    
  if(i==0)       
  {  
   (octaves[0].Octave)[0].levelsigma = initial_sigma;  
   (octaves[0].Octave)[0].absolute_sigma = initial_sigma; 
  }  
  else  
  {  
   (octaves[i].Octave)[0].levelsigma = (octaves[i-1].Octave)[SCALESPEROCTAVE].levelsigma;  
   (octaves[i].Octave)[0].absolute_sigma = (octaves[i-1].Octave)[SCALESPEROCTAVE].absolute_sigma;  
  }  
  sigma = initial_sigma;  
        //建立本阶梯其他层的图像  
  for ( j =  1; j < SCALESPEROCTAVE + 3; j++)   
  {  
   dst = createMat(tempMat->rows, tempMat->cols, XAF_32FC1);//用于存储高斯层  
   temp = createMat(tempMat->rows, tempMat->cols, XAF_32FC1);//用于存储DOG层  
  
   sigma_f= (float)sqrt(k*k-1)*sigma;  
   sigma = (float)k*sigma;  
   absolute_sigma = sigma * (octaves[i].subsample);   
     
   (octaves[i].Octave)[j].levelsigma = sigma;  
   (octaves[i].Octave)[j].absolute_sigma = absolute_sigma;  
            //产生高斯层  
   length=BlurImage(*(octaves[i].Octave)[j-1].Level, dst, sigma_f);//相应尺度  
   (octaves[i].Octave)[j].levelsigmalength = length; 
   (octaves[i].Octave)[j].Level = (xaf_mat**)malloc(sizeof(XAF_MAT*));
   (octaves[i].Octave)[j].Level[0] = dst;
	xafSub( *((octaves[i].Octave)[j]).Level, *((octaves[i].Octave)[j-1]).Level, temp, 0 );
	((DOGoctaves[i].Octave)[j-1]).Level = (xaf_mat**)malloc(sizeof(XAF_MAT*));
	((DOGoctaves[i].Octave)[j-1]).Level[0] = temp;	  
  }  
  // halve the image size for next iteration  
  tempMat  = halfSizeImage( *( (octaves[i].Octave)[SCALESPEROCTAVE].Level ) );  
 }  
 return octaves;//octaves;  
}  

//SIFT算法第三步，特征点位置检测，  
int DetectKeypoint(int numoctaves, ImageOctaves *GaussianPyr)  
{  
   //计算用于DOG极值点检测的主曲率比的阈值  
 double curvature_threshold;  
 int   keypoint_count = 0,i,j,m,n,dim;
 float Dxx,Dyy,Dxy,Tr_H,Det_H,curvature_ratio,inf_val;
 Keypoint k; 
 curvature_threshold= ((CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1))/CURVATURE_THRESHOLD; 
 
 for ( i=0; i<numoctaves; i++)    
 {       
  for(j=1;j<SCALESPEROCTAVE+1;j++)//取中间的scaleperoctave个层  
  {    
   //在图像的有效区域内寻找具有显著性特征的局部最大值  
   //float sigma=(GaussianPyr[i].Octave)[j].levelsigma;  
   //int dim = (int) (max(3.0f, 2.0*GAUSSKERN *sigma + 1.0f)*0.5); 
   dim = (int)(0.5*((GaussianPyr[i].Octave)[j].levelsigmalength)+0.5);
   for (m=dim;m<((DOGoctaves[i].row)-dim);m++)
   {
    for( n=dim;n<((DOGoctaves[i].col)-dim);n++)  
	{	
     if ( abs(ImLevels( DOGoctaves,i,j,m,n))>= CONTRAST_THRESHOLD )  
     {  
		// DOGoctaves[i].Octave[j].Level[0]->data
      if ( ImLevels( DOGoctaves,i,j,m,n)!=0.0 )  //1、首先是非零  
      {  
       inf_val=ImLevels( DOGoctaves,i,j,m,n);  
       if(( (inf_val <= ImLevels( DOGoctaves,i,j-1,m-1,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m  ,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m+1,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m-1,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m  ,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m+1,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m-1,n+1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m  ,n+1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j-1,m+1,n+1))&&    //底层的小尺度9  

        (inf_val <= ImLevels( DOGoctaves,i,j,m-1,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m  ,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m+1,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m-1,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m+1,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m-1,n+1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m  ,n+1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j,m+1,n+1))&&     //当前层8  
          
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m-1,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m  ,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m+1,n-1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m-1,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m  ,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m+1,n  ))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m-1,n+1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m  ,n+1))&&  
        (inf_val <= ImLevels( DOGoctaves,i,j+1,m+1,n+1))     //下一层大尺度9          
        ) ||   
        ( (inf_val >= ImLevels( DOGoctaves,i,j-1,m-1,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m  ,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m+1,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m-1,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m  ,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m+1,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m-1,n+1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m  ,n+1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j-1,m+1,n+1))&&  
          
        (inf_val >= ImLevels( DOGoctaves,i,j,m-1,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m  ,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m+1,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m-1,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m+1,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m-1,n+1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m  ,n+1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j,m+1,n+1))&&   
          
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m-1,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m  ,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m+1,n-1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m-1,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m  ,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m+1,n  ))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m-1,n+1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m  ,n+1))&&  
        (inf_val >= ImLevels( DOGoctaves,i,j+1,m+1,n+1))   
        ) )      //2、满足26个中极值点  
       {     
        //此处可存储  
        //然后必须具有明显的显著性，即必须大于CONTRAST_THRESHOLD=0.02  
        if ( abs(ImLevels( DOGoctaves,i,j,m,n))>= CONTRAST_THRESHOLD )  
        {  
         //最后显著处的特征点必须具有足够的曲率比，CURVATURE_THRESHOLD=10.0，首先计算Hessian矩阵  
         // Compute the entries of the Hessian matrix at the extrema location.  
         /* 
         1   0   -1 
         0   0   0 
         -1   0   1         *0.25 
         */ 
         // Compute the trace and the determinant of the Hessian.  
         //Tr_H = Dxx + Dyy;  
         //Det_H = Dxx*Dyy - Dxy^2;  
           
         Dxx = ImLevels( DOGoctaves,i,j,m,n-1) + ImLevels( DOGoctaves,i,j,m,n+1)-2.0*ImLevels( DOGoctaves,i,j,m,n);  
         Dyy = ImLevels( DOGoctaves,i,j,m-1,n) + ImLevels( DOGoctaves,i,j,m+1,n)-2.0*ImLevels( DOGoctaves,i,j,m,n);  
         Dxy = ImLevels( DOGoctaves,i,j,m-1,n-1) + ImLevels( DOGoctaves,i,j,m+1,n+1) - ImLevels( DOGoctaves,i,j,m+1,n-1) - ImLevels( DOGoctaves,i,j,m-1,n+1);  
         Tr_H = Dxx + Dyy;  
         Det_H = Dxx*Dyy - Dxy*Dxy;  
         // Compute the ratio of the principal curvatures.  
         curvature_ratio = (1.0*Tr_H*Tr_H)/Det_H;  
         if ( (Det_H>=0.0) && (curvature_ratio <= curvature_threshold) )  //最后得到最具有显著性特征的特征点  
         {  
          //将其存储起来，以计算后面的特征描述字  
          keypoint_count++;  
           
          /* Allocate memory for the keypoint. */ 
          k = (Keypoint) malloc(sizeof(struct KeypointSt));  
          k->next = keypoints;  
          keypoints = k;  
          k->row = m*(GaussianPyr[i].subsample);  
          k->col =n*(GaussianPyr[i].subsample);  
          k->sy = m;    //行  
          k->sx = n;    //列  
          k->octave=i;  
          k->level=j;  
          k->scale = (GaussianPyr[i].Octave)[j].absolute_sigma;        
         }//if >curvature_thresh  
        }//if >contrast  
       }//if inf value  
     }//if non zero  
     }//if >contrast  
    }  //for concrete image level col 
   }
  }//for levels 
 }//for octaves  
 return keypoint_count;  
}  
 
//在图像中，显示SIFT特征点的位置  
void DisplayKeypointLocation(int id, ImageOctaves *GaussianPyr TSRMLS_DC)  
{  
 Keypoint p = keypoints; // p指向第一个结点  
 while(p) // 没到表尾  
 {     
  xafLine( id, xafPoint((int)((p->col)-3),(int)(p->row)),xafPoint((int)((p->col)+3),(int)(p->row)), XAF_RGB(255,255,0),1, 8, 0 TSRMLS_CC);  
  xafLine( id, xafPoint((int)(p->col),(int)((p->row)-3)),xafPoint((int)(p->col),(int)((p->row)+3)), XAF_RGB(255,255,0),1, 8, 0 TSRMLS_CC);  
  p=p->next;  
 }   
}  
  
// Compute the gradient direction and magnitude of the gaussian pyramid images  
void ComputeGrad_DirecandMag(int numoctaves, ImageOctaves *GaussianPyr)  
{  
	int i,j,m,n;
 // ImageOctaves *mag_thresh ;  
	mag_pyr=(ImageOctaves*) malloc( numoctaves * sizeof(ImageOctaves) );  
	grad_pyr=(ImageOctaves*) malloc( numoctaves * sizeof(ImageOctaves) );  
 for (i=0; i<numoctaves; i++)    
 {          
	mag_pyr[i].Octave= (ImageLevels*) malloc( (SCALESPEROCTAVE) * sizeof(ImageLevels) );  
	grad_pyr[i].Octave= (ImageLevels*) malloc( (SCALESPEROCTAVE) * sizeof(ImageLevels) );  
  for(j=1;j<SCALESPEROCTAVE+1;j++)//取中间的scaleperoctave个层  
  {    
            XAF_MAT *Mag = createMat(GaussianPyr[i].row, GaussianPyr[i].col, XAF_32FC1);  
   XAF_MAT *Ori = createMat(GaussianPyr[i].row, GaussianPyr[i].col, XAF_32FC1);  
   XAF_MAT *tempMat1 = createMat(GaussianPyr[i].row, GaussianPyr[i].col, XAF_32FC1);  
   XAF_MAT *tempMat2 = createMat(GaussianPyr[i].row, GaussianPyr[i].col, XAF_32FC1);  
   xafZero(Mag);  
   xafZero(Ori);  
   xafZero(tempMat1);  
   xafZero(tempMat2);   
   for ( m=1;m<(GaussianPyr[i].row-1);m++)   
    for( n=1;n<(GaussianPyr[i].col-1);n++)  
    {  
     //计算幅值  
     IMAT(tempMat1,m,n) = 0.5*( ImLevels(DOGoctaves,i,j,m,n+1)-ImLevels(DOGoctaves,i,j,m,n-1) );  //dx  
	 IMAT(tempMat2,m,n) = 0.5*( ImLevels(DOGoctaves,i,j,m+1,n)-ImLevels(DOGoctaves,i,j,m-1,n) );  //dy  
	 IMAT(Mag,m,n) = sqrt(IMAT(tempMat1,m,n)*IMAT(tempMat1,m,n)+IMAT(tempMat2,m,n)*IMAT(tempMat2,m,n));  //mag  
     //计算方向  
     IMAT(Ori,m,n) =atan( IMAT(tempMat2,m,n)/IMAT(tempMat1,m,n) );  
     if (IMAT(Ori,m,n)==PI)  
      IMAT(Ori,m,n)=-PI;  
    }  
	((mag_pyr[i].Octave)[j-1]).Level = (XAF_MAT**)malloc(sizeof(XAF_MAT*));
	((grad_pyr[i].Octave)[j-1]).Level = (XAF_MAT**)malloc(sizeof(XAF_MAT*));
    ((mag_pyr[i].Octave)[j-1]).Level=&Mag;  
    ((grad_pyr[i].Octave)[j-1]).Level=&Ori;  
    releaseMat(&tempMat1);  
    releaseMat(&tempMat2);  
  }//for levels  
 }//for octaves  
}   

//SIFT算法第四步：计算各个特征点的主方向，确定主方向  
void AssignTheMainOrientation(int numoctaves, ImageOctaves *GaussianPyr,ImageOctaves *mag_pyr,ImageOctaves *grad_pyr)  
{  
    // Set up the histogram bin centers for a 36 bin histogram.  
    int num_bins = 36,zero_pad,keypoint_count,i,j,m,n,dim,sn,b,sw,y,nn,mm,x,binIdx,ci;  
    float hist_step = 2.0*PI/num_bins,sigma1,sigma;
	double *orienthist,dy,dx,mag;double Ori;
	double maxGrad = 0.0;  
	int maxBin = 0;
	bool binIsKeypoint[36];
	Keypoint p;
	Keypoint k; 
	double oneBinRad = (2.0 * PI) / 36;
	int leftI,rightI;
	int bLeft,bRight;
	double peakValue;  
	double degreeCorrection;  
	double maxPeakValue = 0.0, maxDegreeCorrection = 0.0,degree; 
	XAF_MAT* mat;
	sigma1=( ((GaussianPyr[0].Octave)[SCALESPEROCTAVE].absolute_sigma) ) / (GaussianPyr[0].subsample);//SCALESPEROCTAVE+2  
	zero_pad = (int) (max(3.0f, 2 * GAUSSKERN *sigma1 + 1.0f)*0.5+0.5);  
    //Assign orientations to the keypoints.  
	keypoint_count = 0;  
	p = keypoints; // p指向第一个结点  
   ci = 0;
 while(p) // 没到表尾  
 {  ci++;
    i=p->octave;  
    j=p->level;  
    m=p->sy;   //行  
    n=p->sx;   //列  
	
	if((m>=zero_pad)&&(m<GaussianPyr[i].row-zero_pad)&&(n>=zero_pad)&&(n<GaussianPyr[i].col-zero_pad) )  
	{  
		sigma=( ((GaussianPyr[i].Octave)[j].absolute_sigma) ) / (GaussianPyr[i].subsample);  
		//产生二维高斯模板  
		mat = GaussianKernel2D( sigma );           
		dim=(int)(0.5 * (mat->rows));  
		//分配用于存储Patch幅值和方向的空间  
     
		//声明方向直方图变量  
		orienthist = (double *) malloc(36 * sizeof(double));  
	   for (sw = 0 ; sw < 36 ; ++sw)   
	   {  
		orienthist[sw]=0.0;    
	   } 
	   
		//在特征点的周围统计梯度方向  
		for (x=m-dim,mm=0;x<=(m+dim);x++,mm++)
		{
			for(y=n-dim,nn=0;y<=(n+dim);y++,nn++)  
			{  
				if(x<=0||y<=0||(x+1) >= DOGoctaves[i].Octave[j].Level[0]->rows||(y+1) >= DOGoctaves[i].Octave[j].Level[0]->cols )
				   continue;
			   dx = 0.5*(ImLevels( DOGoctaves,i,j,x,y+1)-ImLevels( DOGoctaves,i,j,x,y-1));   
			   dy = 0.5*(ImLevels( DOGoctaves,i,j,x+1,y)-ImLevels( DOGoctaves,i,j,x-1,y));   
			   mag = sqrt(dx*dx+dy*dy);  
			 
			   Ori =atan( 1.0*dy/dx );  
			   binIdx = FindClosestRotationBin(36, Ori);               
			
			   orienthist[binIdx] = orienthist[binIdx] + 1.0* mag * IMAT(mat,mm,nn);
			
			}
		}

    // Find peaks in the orientation histogram using nonmax suppression.
	 
    AverageWeakBins (orienthist, 36); 
    // find the maximum peak in gradient orientation  
	{
		 
		for (b = 0 ; b < 36 ; ++b)   
		{  
		 if (orienthist[b] > maxGrad)   
		 {  
		  maxGrad = orienthist[b];  
		  maxBin = b;  
		 }  
		}
	}
    // First determine the real interpolated peak high at the maximum bin  
    // position, which is guaranteed to be an absolute peak.
	  
    if ( (InterpolateOrientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
                    orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
                    &maxDegreeCorrection, &maxPeakValue)) == false)  

   
    for ( b = 0 ; b < 36 ; ++b)   
    {  
     binIsKeypoint[b] = false;  
     // The maximum peak of course is  
     if (b == maxBin)   
     {  
      binIsKeypoint[b] = true;  
      continue;  
     }  
     // Local peaks are, too, in case they fulfill the threshhold  
     if (orienthist[b] < (peakRelThresh * maxPeakValue))  
      continue;  
       leftI = (b == 0) ? (36 - 1) : (b - 1);  
       rightI = (b + 1) % 36;  
     if (orienthist[b] <= orienthist[leftI] || orienthist[b] <= orienthist[rightI])  
      continue; // no local peak  
     binIsKeypoint[b] = true;  
    }  
    // find other possible locations  
  
    for ( b = 0 ; b < 36 ; ++b)   
    {  
     if (binIsKeypoint[b] == false)  
      continue;  
       bLeft = (b == 0) ? (36 - 1) : (b - 1);  
       bRight = (b + 1) % 36;  
     // Get an interpolated peak direction and value guess.                  
     if (InterpolateOrientation ( orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],  
      orienthist[maxBin], orienthist[(maxBin + 1) % 36],  
      &degreeCorrection, &peakValue) == false)  
     {  
      //printf("BUG: Parabola fitting broken");  
		 continue;
     }  
       
       degree = (b + degreeCorrection) * oneBinRad - PI;  
     if (degree < -PI)  
      degree += 2.0 * PI;  
     else if (degree > PI)  
      degree -= 2.0 * PI;  
     //存储方向，可以直接利用检测到的链表进行该步主方向的指定;  
     //分配内存重新存储特征点  
      
     ///-* Allocate memory for the keypoint Descriptor. *-/  
     k = (Keypoint) malloc(sizeof(struct KeypointSt));  
     k->next = keyDescriptors;  
     keyDescriptors = k;  
     k->descrip = (float*)malloc(LEN * sizeof(float));  
     k->row = p->row;  
     k->col = p->col;  
     k->sy = p->sy;    //行  
     k->sx = p->sx;    //列  
     k->octave = p->octave;  
     k->level = p->level;  
     k->scale = p->scale;        
     k->ori = degree;  
     k->mag = peakValue;   
    }//for
    free(orienthist);  
  } 
  p =p->next;  
 }   
}  
  
//寻找与方向直方图最近的柱，确定其index   
int FindClosestRotationBin (int binCount, float angle)  
{
 int idx;
 angle += PI;  
 angle /= 2.0 * PI;  
 // calculate the aligned bin  
 angle *= binCount;  
 idx = (int) angle;  
 if (idx == binCount)  
	idx = 0;  
 return (idx);  
}  
  
// Average the content of the direction bins.  
void AverageWeakBins (double* hist, int binCount)  
{  
 // TODO: make some tests what number of passes is the best. (its clear  
 // one is not enough, as we may have something like  
 // ( 0.4, 0.4, 0.3, 0.4, 0.4 )) 
	int sn,sw;
	double firstE = hist[0];  
	double last = hist[binCount-1],cur,next;
 for (  sn = 0 ; sn < 2 ; ++sn)   
 {  
    
  for (  sw = 0 ; sw < binCount ; ++sw)   
  {  
     cur = hist[sw];  
     next = (sw == (binCount - 1)) ? firstE : hist[(sw + 1) % binCount];  
     hist[sw] = (last + cur + next) / 3.0;  
     last = cur;  
  }  
 }  
}  
  
// Fit a parabol to the three points (-1.0 ; left), (0.0 ; middle) and  
// (1.0 ; right).  
// Formulas:  
// f(x) = a (x - c)^2 + b  
// c is the peak offset (where f'(x) is zero), b is the peak value.  
// In case there is an error false is returned, otherwise a correction  
// value between [-1 ; 1] is returned in 'degreeCorrection', where -1  
// means the peak is located completely at the left vector, and -0.5 just  
// in the middle between left and middle and > 0 to the right side. In  
// 'peakValue' the maximum estimated peak value is stored.  
bool InterpolateOrientation (double left, double middle,double right, double *degreeCorrection, double *peakValue)  
{  
 double a = ((left + right) - 2.0 * middle) / 2.0;   //抛物线捏合系数a  
 double c,b;
 // degreeCorrection = peakValue = Double.NaN;  
   
 // Not a parabol  
 if (a == 0.0)  
  return false;  
   c = (((left - middle) / a) - 1.0) / 2.0;  
   b = middle - c * c * a;  
 if (c < -0.5 || c > 0.5)  
  return false;  
 *degreeCorrection = c;  
 *peakValue = b;  
 return true;  
}  
  
//显示特征点处的主方向  
void DisplayOrientation (int image, ImageOctaves *GaussianPyr TSRMLS_DC)  
{
	float scale,autoscale = 3.0f,uu,vv,x,y,alpha = 0.33,beta = 0.33,xx0,yy0,xx1,yy1;
	Keypoint p = keyDescriptors; // p指向第一个结点  
 while(p) // 没到表尾  
 {  
    scale=(GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;   
    uu=autoscale*scale*cos(p->ori);  
    vv=autoscale*scale*sin(p->ori);  
    x=(p->col)+uu;  
    y=(p->row)+vv;  
    xafLine( image, xafPoint((int)(p->col),(int)(p->row)),xafPoint((int)x,(int)y), XAF_RGB(255,255,0),1, 8, 0 TSRMLS_CC);  
  // Arrow head parameters  
    xx0= (p->col)+uu-alpha*(uu+beta*vv);  
    yy0= (p->row)+vv-alpha*(vv-beta*uu);  
    xx1= (p->col)+uu-alpha*(uu-beta*vv);  
    yy1= (p->row)+vv-alpha*(vv+beta*uu);  
    xafLine( image, xafPoint((int)xx0,(int)yy0),xafPoint((int)x,(int)y), XAF_RGB(255,255,0),1, 8, 0 TSRMLS_CC);  
    xafLine( image, xafPoint((int)xx1,(int)yy1),xafPoint((int)x,(int)y), XAF_RGB(255,255,0),1, 8, 0 TSRMLS_CC);  
    p=p->next;  
 }   
} 

void ExtractFeatureDescriptors(int numoctaves, ImageOctaves *GaussianPyr)  
{  
 // The orientation histograms have 8 bins  
 int m,i,j,x,y,n;
 float orient_bin_spacing = PI/4;  
 float orient_angles[8]={-PI,-PI+orient_bin_spacing,-PI*0.5, -orient_bin_spacing,0.0, orient_bin_spacing, PI*0.5,  PI+orient_bin_spacing};  
    //产生描述字中心各点坐标  
 float *feat_grid=(float *) malloc( 2*16 * sizeof(float)),*feat_samples,feat_window;
 float scale,sine,cosine,*featcenter,*feat,*feat_desc,x_sample,y_sample;
 float sample12,sample21,sample22,sample23,sample32;   
   //float diff_x = 0.5*(sample23 - sample21);  
            //float diff_y = 0.5*(sample32 - sample12);  
 float diff_x ,diff_y,mag_sample,grad_sample,*x_wght,*y_wght,*pos_wght; 
 float xf,yf,g;float diff[8],orient_wght[128],angle,temp,norm;
 Keypoint p ;
 for (i=0;i<GridSpacing;i++)  
 {  
  for (j=0;j<2*GridSpacing;++j,++j)  
  {  
   feat_grid[i*2*GridSpacing+j]=-6.0+i*GridSpacing;  
   feat_grid[i*2*GridSpacing+j+1]=-6.0+0.5*j*GridSpacing;  
  }  
 }  

    //产生网格  
feat_samples=(float *) malloc( 2*256 * sizeof(float));  
for ( i=0;i<4*GridSpacing;i++)  
 {  
	  for (j=0;j<8*GridSpacing;j+=2)  
	  {  
	   feat_samples[i*8*GridSpacing+j]=-(2*GridSpacing-0.5)+i;  
	   feat_samples[i*8*GridSpacing+j+1]=-(2*GridSpacing-0.5)+0.5*j;  
	  }  
 } 

 feat_window = 2*GridSpacing;  
 p = keyDescriptors; // p指向第一个结点  
 while(p) // 没到表尾  
 {  
  float scale=(GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;  
          
    sine = sin(p->ori);  
    cosine = cos(p->ori);    
        //计算中心点坐标旋转之后的位置  
   featcenter=(float *) malloc( 2*16 * sizeof(float));  
  for (i=0;i<GridSpacing;i++)  
  {
   for (  j=0;j<2*GridSpacing;j+=2)  
   {  
      x=feat_grid[i*2*GridSpacing+j];  
      y=feat_grid[i*2*GridSpacing+j+1];  
    featcenter[i*2*GridSpacing+j]=((cosine * x + sine * y) + p->sx);  
    featcenter[i*2*GridSpacing+j+1]=((-sine * x + cosine * y) + p->sy);  
   }  
  }  
  // calculate sample window coordinates (rotated along keypoint)  
  feat=(float *) malloc( 2*256 * sizeof(float));  
  for ( i=0;i<64*GridSpacing;i++,i++)  
  {  
     x=feat_samples[i];  
     y=feat_samples[i+1];  
   feat[i]=((cosine * x + sine * y) + p->sx);  
   feat[i+1]=((-sine * x + cosine * y) + p->sy);  
  }  
  //Initialize the feature descriptor.  
   feat_desc = (float *) malloc( 128 * sizeof(float));  
  for (i=0;i<128;i++)  
  {  
	feat_desc[i]=0.0;  
   // printf("%f  ",feat_desc[i]);    
  }  
        //printf("/n");  
  for ( i=0;i<512;++i,++i)  
  {  
	  x_sample = feat[i];  
	  y_sample = feat[i+1];  
            // Interpolate the gradient at the sample position  
      //* 
     // 0   1   0 
    //  1   *   1 
    //  0   1   0   具体插值策略如图示 
  // */  /*
     sample12=getPixelBI(*((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample-1);  

     sample21=getPixelBI(*((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample-1, y_sample);   
     sample22=getPixelBI(*((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample);   
     sample23=getPixelBI(*((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample+1, y_sample);   
     sample32=getPixelBI(*((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample+1);   
   //float diff_x = 0.5*(sample23 - sample21);  
            //float diff_y = 0.5*(sample32 - sample12);  
     diff_x = sample23 - sample21;  
              diff_y = sample32 - sample12;  
              mag_sample = sqrt( diff_x*diff_x + diff_y*diff_y );  
              grad_sample = atan( diff_y / diff_x );  
            if(grad_sample == PI)  
				grad_sample = -PI;  
            // Compute the weighting for the x and y dimensions.  
              x_wght=(float *) malloc( GridSpacing * GridSpacing * sizeof(float));  
              y_wght=(float *) malloc( GridSpacing * GridSpacing * sizeof(float));  
              pos_wght=(float *) malloc( 8*GridSpacing * GridSpacing * sizeof(float));;  
   for (  m=0;m<32;++m,++m)  
   {  
      x=featcenter[m];  
      y=featcenter[m+1];  
    x_wght[m/2] = max(1 - (abs(x - x_sample)*1.0/GridSpacing), 0);  
                y_wght[m/2] = max(1 - (abs(y - y_sample)*1.0/GridSpacing), 0);   
      
   }  
   for ( m=0;m<16;++m)  
      for (  n=0;n<8;++n)  
		pos_wght[m*8+n]=x_wght[m]*y_wght[m];  
    free(x_wght);  
    free(y_wght);  
    //计算方向的加权，首先旋转梯度场到主方向，然后计算差异   
      
    for ( m=0;m<8;++m)  
    {   
       angle = grad_sample-(p->ori)-orient_angles[m]+PI;  
       temp = angle / (2.0 * PI);  
     angle -= (int)(temp) * (2.0 * PI);  
     diff[m]= angle - PI;  
    }  
    // Compute the gaussian weighting.  
      xf=p->sx;  
      yf=p->sy;  
      g = exp(-((x_sample-xf)*(x_sample-xf)+(y_sample-yf)*(y_sample-yf))/(2*feat_window*feat_window))/(2*PI*feat_window*feat_window);  
      
    for ( m=0;m<128;++m)  
    {  
     orient_wght[m] = max((1.0 - 1.0*abs(diff[m%8])/orient_bin_spacing),0);  
     feat_desc[m] = feat_desc[m] + orient_wght[m]*pos_wght[m]*g*mag_sample;  
    }  
    free(pos_wght);     
  }  
  free(feat);  
  free(featcenter); 
    norm=GetVecNorm( feat_desc, 128);  
  for (m=0;m<128;m++)  
  {  
   feat_desc[m]/=norm;  
   if (feat_desc[m]>0.2)  
    feat_desc[m]=0.2;  
  }  
        norm=GetVecNorm( feat_desc, 128);  
        for ( m=0;m<128;m++)  
  {  
   feat_desc[m]/=norm;    
  }   
  p->descrip = feat_desc;  
  p=p->next;  
 }  

 free(feat_grid);  
 free(feat_samples);  
}  
  
//为了显示图象金字塔，而作的图像水平拼接  
XAF_MAT* MosaicHorizen( XAF_MAT* im1, XAF_MAT* im2 )  
{  
 int row,col;  
 XAF_MAT *mosaic = createMat( max(im1->rows,im2->rows),(im1->cols+im2->cols),XAF_32FC1);  
xafZero(mosaic);  
 /* Copy images into mosaic1. */  
 for ( row = 0; row < im1->rows; row++)  
  for ( col = 0; col < im1->cols; col++)  
	  IMAT(mosaic,row,col)=IMAT(im1,row,col) ;  
  for (  row = 0; row < im2->rows; row++)  
   for (  col = 0; col < im2->cols; col++)  
    IMAT(mosaic,row,(col+im1->cols) )= IMAT(im2,row,col) ;  
   return mosaic;  
}  
  
//为了显示图象金字塔，而作的图像垂直拼接  
XAF_MAT* MosaicVertical( XAF_MAT* im1, XAF_MAT* im2 )  
{  
 int row,col;  
 XAF_MAT *mosaic = createMat(im1->rows+im2->rows,max(im1->cols,im2->cols), XAF_32FC1);  
 xafZero(mosaic);  
   
 for ( row = 0; row < im1->rows; row++)  
  for ( col = 0; col < im1->cols; col++)  
	  IMAT(mosaic,row,col)= IMAT(im1,row,col) ;  
  for ( row = 0; row < im2->rows; row++)  
   for ( col = 0; col < im2->cols; col++)  
    IMAT(mosaic,(row+im1->rows),col)=IMAT(im2,row,col) ;  
     
   return mosaic;  
}  

int test(gdImagePtr im,int id TSRMLS_DC)  
{  
 //声明当前帧IplImage指针  
 gdImagePtr src = NULL;   
 gdImagePtr grey_im1 = NULL;   
 gdImagePtr DoubleSizeImage = NULL;  
 int i,j,grey_im1_id,DoubleSizeImage_id;
 XAF_MAT* image1Mat = NULL;  
 XAF_MAT* tempMat=NULL;  
   
 ImageOctaves *Gaussianpyr;  
 int rows,cols,dim,keycount;  
 src = im;    
 grey_im1 = CREATEIMAGE(gdImageSX(src), gdImageSY(src),&grey_im1_id );  
 DoubleSizeImage = CREATEIMAGE(2*gdImageSX(src), 2*gdImageSY(src) ,&DoubleSizeImage_id);  

 //为图像阵列分配内存，假设两幅图像的大小相同，tempMat跟随image1的大小  
 image1Mat = createMat(gdImageSY(src), gdImageSX(src), XAF_32FC1);  
 //转化成单通道图像再处理 
 COPYIMAGE(grey_im1_id,id,src,grey_im1);
  
//grayImage(grey_im1);  
 //转换进入Mat数据结构,图像操作使用的是浮点型操作 
 xafConvert(grey_im1, image1Mat); 
 //图像归一化  
xafConvertScale( image1Mat, image1Mat, 1.0/255,0 );
 
 dim = min(image1Mat->rows, image1Mat->cols);  
 numoctaves = (int) (log((double) dim) / log(2.0)) - 2;    //金字塔阶数  
 numoctaves = min(numoctaves, MAXOCTAVES);  

 //SIFT算法第一步，预滤波除噪声，建立金字塔底层 

tempMat = cloneMat(image1Mat); 
 BlurImage(image1Mat, tempMat, INITSIGMA);  
 tempMat = ScaleInitImage(image1Mat) ;  

  xafConvert2Image(tempMat,DoubleSizeImage);
 //SIFT算法第二步，建立Guassian金字塔和DOG金字塔  
 Gaussianpyr = BuildGaussianOctaves(tempMat) ;
 
 //SIFT算法第三步：特征点位置检测，最后确定特征点的位置        
  keycount=DetectKeypoint(numoctaves, Gaussianpyr);
  DisplayKeypointLocation( id ,Gaussianpyr TSRMLS_CC); 
 //SIFT算法第四步：计算高斯图像的梯度方向和幅值，计算各个特征点的主方向  
 ComputeGrad_DirecandMag(numoctaves, Gaussianpyr);
 AssignTheMainOrientation( numoctaves, Gaussianpyr,mag_pyr,grad_pyr);  
 DisplayOrientation ( id, Gaussianpyr TSRMLS_CC);  
 //SIFT算法第五步：抽取各个特征点处的特征描述字  
 ExtractFeatureDescriptors( numoctaves, Gaussianpyr); 

  xafimagedestroy(grey_im1_id TSRMLS_CC);
  xafimagedestroy(DoubleSizeImage_id TSRMLS_CC);
  releaseMat(&image1Mat);
  releaseMat(&tempMat);
 return DoubleSizeImage_id;  
}  

KeypointPtr* siftArray()
{
	KeypointPtr *p;
	p= keyDescriptors;
	return p;
}
