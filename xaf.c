/*
  +----------------------------------------------------------------------+
  | PHP Version 5                                                        |
  +----------------------------------------------------------------------+
  | Copyright (c) 1997-2013 The PHP Group                                |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author:                                                              |
  +----------------------------------------------------------------------+
*/

/* $Id$ */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "ext/gd/php_gd.h"
#include "ext/gd/libgd/gd.h"
#include "Zend/zend_interfaces.h"
#include "sift.h"
#include "php_xaf.h"
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
/*PHPAPI int phpi_get_le_gd();*/
/* If you declare any globals in php_xaf.h uncomment this:
ZEND_DECLARE_MODULE_GLOBALS(xaf)
*/
/*
static long  colorPoint(gdImagePtr im,int x,int y);
static int colorChange(gdImagePtr im,int color,int x,int y);
static void grayImage(gdImagePtr im);
static double **transposingMatrix( double **mat, int n);
static double** coefficient(int n);
static double** matrixMultiply( double** A, double** B, int n);
static int** Dct( double** iMatrix, int n);
static int averageGray(int** pix, int w, int h);
*/
/* True global resources - no need for thread safety here */
static int le_xaf,le_gd;

/* {{{ xaf_functions[]
 *
 * Every user visible function must have an entry in xaf_functions[].
 */

ZEND_BEGIN_ARG_INFO(arginfo_xaf_imgDct, 0)
	ZEND_ARG_INFO(0, im)
	ZEND_ARG_INFO(0, x)
	ZEND_ARG_INFO(0, y)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO(arginfo_xaf_hmdist, 0)
	ZEND_ARG_INFO(0, a)
	ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO(arginfo_xaf_sift, 0)
	ZEND_ARG_INFO(0, im)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO(arginfo_xaf_gray, 0)
	ZEND_ARG_INFO(0, im)
ZEND_END_ARG_INFO()

const zend_function_entry xaf_functions[] = {
	PHP_FE(confirm_xaf_compiled,	NULL)		/* For testing, remove later. */
	PHP_FE(xaf_rtype,	NULL)
	PHP_FE(xaf_sift,	arginfo_xaf_sift)
	PHP_FE(xaf_gray,	arginfo_xaf_gray)
    PHP_FE(xaf_imgDct,arginfo_xaf_imgDct)
	PHP_FE(xaf_hmdist,arginfo_xaf_hmdist)
	PHP_FE_END	/* Must be the last line in xaf_functions[] */
};

static zend_module_dep pdo_mysql_deps[] = {
    ZEND_MOD_REQUIRED("gd2")
    {NULL, NULL, NULL}
};

/* }}} */

/* {{{ xaf_module_entry
 */

zend_module_entry xaf_module_entry = {
#if ZEND_MODULE_API_NO >= 20010901
	STANDARD_MODULE_HEADER,
#endif
	"xaf",
	xaf_functions,
	PHP_MINIT(xaf),
	PHP_MSHUTDOWN(xaf),
	PHP_RINIT(xaf),		/* Replace with NULL if there's nothing to do at request start */
	PHP_RSHUTDOWN(xaf),	/* Replace with NULL if there's nothing to do at request end */
	PHP_MINFO(xaf),
#if ZEND_MODULE_API_NO >= 20010901
	"0.1", /* Replace with version number for your extension */
#endif
	STANDARD_MODULE_PROPERTIES
};
/* }}} */

#ifdef COMPILE_DL_XAF
ZEND_GET_MODULE(xaf)
#endif

/* {{{ PHP_INI
 */
/* Remove comments and fill if you need to have entries in php.ini
PHP_INI_BEGIN()
    STD_PHP_INI_ENTRY("xaf.global_value",      "42", PHP_INI_ALL, OnUpdateLong, global_value, zend_xaf_globals, xaf_globals)
    STD_PHP_INI_ENTRY("xaf.global_string", "foobar", PHP_INI_ALL, OnUpdateString, global_string, zend_xaf_globals, xaf_globals)
PHP_INI_END()
*/
/* }}} */

/* {{{ php_xaf_init_globals
 */
/* Uncomment this function if you have INI entries
static void php_xaf_init_globals(zend_xaf_globals *xaf_globals)
{
	xaf_globals->global_value = 0;
	xaf_globals->global_string = NULL;
}
*/
/* }}} */

/* {{{ PHP_MINIT_FUNCTION
 */
PHP_MINIT_FUNCTION(xaf)
{
	/* If you have INI entries, uncomment these lines 
	REGISTER_INI_ENTRIES();
	*/
	le_gd = phpi_get_le_gd();
	return SUCCESS;
}
/* }}} */

/* {{{ PHP_MSHUTDOWN_FUNCTION
 */
PHP_MSHUTDOWN_FUNCTION(xaf)
{
	/* uncomment this line if you have INI entries
	UNREGISTER_INI_ENTRIES();
	*/
	return SUCCESS;
}

/* }}} */

/* Remove if there's nothing to do at request start */
/* {{{ PHP_RINIT_FUNCTION
 */
PHP_RINIT_FUNCTION(xaf)
{
	return SUCCESS;
}
/* }}} */

/* Remove if there's nothing to do at request end */
/* {{{ PHP_RSHUTDOWN_FUNCTION
 */
PHP_RSHUTDOWN_FUNCTION(xaf)
{
	return SUCCESS;
}
/* }}} */

/* {{{ PHP_MINFO_FUNCTION
 */
PHP_MINFO_FUNCTION(xaf)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "xaf support", "enabled");
	
	php_info_print_table_header(2, "xaf_imgDct support", "enabled");
	php_info_print_table_end();

	/* Remove comments if you have entries in php.ini
	DISPLAY_INI_ENTRIES();
	*/
}
/* }}} */


/* Remove the following function when you have succesfully modified config.m4
   so that your module can be compiled into PHP, it exists only for testing
   purposes. */

/* Every user-visible function in PHP should document itself in the source */
/* {{{ proto string confirm_xaf_compiled(string arg)
   Return a string to confirm that the module is compiled in */

long  colorPoint(gdImagePtr im,int x,int y)
{
	if (gdImageTrueColor(im)) {
		if (im->tpixels && gdImageBoundsSafe(im, x, y)) {
			return gdImageTrueColorPixel(im, x, y);
		} else {
			
			return 0;
		}
	} else {
		if (im->pixels && gdImageBoundsSafe(im, x, y)) {
			return im->pixels[y][x];
		} else {

			return 0;
		}
	}
}

int colorChange(gdImagePtr im,int color,int x,int y)
{
	if (gdImageTrueColor(im)) {
		if (im->tpixels && gdImageBoundsSafe(im, x, y)) {
			gdImageTrueColorPixel(im, x, y) = color;
			return 1;
		} else {
			
			return 0;
		}
	} else {
		if (im->pixels && gdImageBoundsSafe(im, x, y)) {
			im->pixels[y][x] = color;
			return 1;
		} else {

			return 0;
		}
	}
}

void grayImage(gdImagePtr im) 
{  
	int color,point,r,g,b,h,w;
	h = gdImageSY(im);
	w = gdImageSX(im);
	{
		int i,j;
        for(i=0; i<h; i++) {  
            for(j=0; j<w; j++) {  
           
				point = colorPoint(im,j,i);
				r = gdImageRed(im,point);
				g = gdImageGreen(im,point);
				b = gdImageBlue(im,point);
                color =  (cscGr_32f*r + cscGg_32f*g + cscGb_32f*b );  
				colorChange(im,color,j,i);
            }  
        }
	}

}


static double** transposingMatrix( double **mat,int n)
{
	double **Matrix = (double**)emalloc(n*sizeof(double*));  
	VALARR_INIT(Matrix,n,n,double);
	{
		int i,j;
        for(i=0; i<n; i++) {  
            for(j=0; j<n; j++) {  
                Matrix[i][j] = mat[j][i];  
            }  
        }  
	}
        return Matrix;  
}

static double** coefficient(int n) { 

		double **coeff = (double**)emalloc(n*sizeof(double*)); 
		double Sqrt;
		VALARR_INIT(coeff,n,n,double);

		Sqrt = 1.0/sqrt((double)n);  
	{
		int i,j;
        for(i=0; i<n; i++) {  
            coeff[0][i] = Sqrt;  
        }  
        for(i=1; i<n; i++) {  
            for(j=0; j<n; j++) {  
                coeff[i][j] = sqrt(2.0/n) * cos(i*PI*(j+0.5)/(double)n);  
            }  
        }  
	}
        return coeff;  
}  

 static double** matrixMultiply( double** A, double** B, int n) {  
		double **nMatrix = (double**)emalloc(n*sizeof(double*)); 
        double t = 0.0; 
		int i,j,k;
		VALARR_INIT(nMatrix,n,n,double);
        for(i=0; i<n; i++) {  
            for(j=0; j<n; j++) {  
                t = 0;  
                for(k=0; k<n; k++) {  
                    t += A[i][k]*B[k][j];  
                }  
                nMatrix[i][j] = t;          
			}  
        }  
        return nMatrix;  
} 

 static unsigned int** Dct( double** iMatrix, int n,gdImagePtr im) {          
       
        double** quotient = coefficient(n);    
        double** quotientT;
		double** imT; 
        double** temp = NULL;
		unsigned int** newpix = (unsigned int**)emalloc(sizeof(unsigned int*)*8);
		int i,j;double ave;

        
       
		ave = average(iMatrix,n,n);
		for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			iMatrix[i][j]>ave ? colorChange(im,XAF_RGB(255,255,255),i,j):colorChange(im,0,i,j);
			
		}
		}
		quotientT = transposingMatrix(quotient, n);
		imT = transposingMatrix(iMatrix, n);
		temp = matrixMultiply(quotient, iMatrix, n); 
		iMatrix = matrixMultiply(temp, quotientT, n);

		VALARR_INIT(newpix,8,8,unsigned int);
        for(i=0; i<8; i++) {  
            for(j=0; j<8; j++) {  
                newpix[i][j] = (int)iMatrix[i][j];  
            }  
        }
		VALARR_FREE(quotient,n);
		VALARR_FREE(quotientT,n);
		VALARR_FREE(temp,n)
		VALARR_FREE(iMatrix,n);
		efree(quotient);
		efree(quotientT);
		efree(temp);
		efree(iMatrix);
        return newpix;  
}  

static unsigned int averageGray(unsigned int** pix, int w, int h) {  
        unsigned int sum = 0;
		int i,j,count = 0;  
        for( i=0; i<h; i++) {  
            for(j=0; j<w; j++) {  
                sum = sum+pix[i][j]; 
				if(pix[i][j]){
					count++;
				}
            }  
              
        }  
        return (unsigned int)(sum/(count));  
} 
static  double average(double** pix, int w, int h) {  
        double sum = 0;
		int i,j,count = 0;  
        for( i=0; i<h; i++) {  
            for(j=0; j<w; j++) {  
                sum = sum+pix[i][j]; 
				if(pix[i][j]){
					count++;
				}
            }  
              
        }  
        return (double)(sum/(count));  
} 
static char hexconvtab[] = "0123456789abcdef";
static char *xaf_bin2hex(const unsigned char *old, const int oldlen, int *newlen)
{
	register unsigned char *result = NULL;
	int i, j;

	result = (unsigned char *) safe_emalloc(oldlen, 2 * sizeof(char), 1);

	for (i = j = 0; i < oldlen; i++) {
		result[j++] = hexconvtab[old[i] >> 4];
		result[j++] = hexconvtab[old[i] & 15];
	}
	result[j] = '\0';

	if (newlen)
		*newlen = oldlen * 2 * sizeof(char);

	return (char *)result;
}

static char *xaf_hex2bin(const unsigned char *old, const size_t oldlen, size_t *newlen)
{
	size_t target_length = oldlen >> 1;
	register unsigned char *str = (unsigned char *)safe_emalloc(target_length, sizeof(char), 1);
	size_t i, j;
	for (i = j = 0; i < target_length; i++) {
		char c = old[j++];
		if (c >= '0' && c <= '9') {
			str[i] = (c - '0') << 4;
		} else if (c >= 'a' && c <= 'f') {
			str[i] = (c - 'a' + 10) << 4;
		} else if (c >= 'A' && c <= 'F') {
			str[i] = (c - 'A' + 10) << 4;
		} else {
			efree(str);
			return NULL;
		}
		c = old[j++];
		if (c >= '0' && c <= '9') {
			str[i] |= c - '0';
		} else if (c >= 'a' && c <= 'f') {
			str[i] |= c - 'a' + 10;
		} else if (c >= 'A' && c <= 'F') {
			str[i] |= c - 'A' + 10;
		} else {
			efree(str);
			return NULL;
		}
	}
	str[target_length] = '\0';

	if (newlen)
		*newlen = target_length;

	return (char *)str;
}

static int flag[] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
PHP_FUNCTION(confirm_xaf_compiled)
{
	char *arg = NULL;
	int arg_len, len;
	char *strg;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "s", &arg, &arg_len) == FAILURE) {
		return;
	}

	len = spprintf(&strg, 0, "Congratulations! You have successfully modified ext/%.78s/config.m4. Module %.78s is now compiled into PHP.", "xaf", arg);
	RETURN_STRINGL(strg, len, 0);

}

PHP_FUNCTION(xaf_rtype)
{
	
	RETURN_LONG(phpi_get_le_gd());
/*	RETURN_LONG(100000);*/
}

PHP_FUNCTION(xaf_imgDct)
{
	zval *IM;
	gdImagePtr im;
	double **iMatrix;
	unsigned int **pix,ave=0;int n,i,j;
	char bit[BITNSLOTS(64)];
	int len;
	char *str;
	memset(bit, 0, BITNSLOTS(64));
	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "rl", &IM,&n) == FAILURE) {
		return;
	}
	
	ZEND_FETCH_RESOURCE(im, gdImagePtr, &IM, -1, "Image",le_gd );
	grayImage(im);
	iMatrix = (double**)emalloc(n*sizeof(double*));  
	VALARR_INIT(iMatrix,n,n,double);

	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			iMatrix[j][i] = colorPoint(im,j,i);
		}
	}
	
	pix = Dct(iMatrix,n,im);

	ave = averageGray(pix,8,8);
	
	for(i=0;i<8;i++){
		for(j=0;j<8;j++){

			if(pix[i][j] >= ave )
			{
				BITSET(bit,(i*8+j));
			}else{
				continue;
			}
			
		}
		
	}
	VALARR_FREE(iMatrix,n);
	VALARR_FREE(pix,8);
	efree(pix);
	efree(iMatrix);
	str = xaf_bin2hex((unsigned char*)bit,8,&len);
	RETURN_STRINGL(str,len,0);

}

PHP_FUNCTION(xaf_hmdist)
{
	char *a,*b;
	int al,bl,len,i;
	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "ss", &a,&al,&b,&bl) == FAILURE) {
		return;
	}

	a = xaf_hex2bin((unsigned char*)a,al,(size_t*)&len);
	b = xaf_hex2bin((unsigned char*)b,bl,(size_t*)&len);
	*(long long*)a ^= *(long long*)b;
	len = 0;
	for (i = 0; i < 8; i++) { 
		   len += flag[a[i]&0xf]; 
		   len += flag[a[i]>>4&0xf];
	}  

	RETURN_LONG(len);
}
PHP_FUNCTION(xaf_sift)
{
	gdImagePtr im;
	gdImagePtr newim;
	zval *IM,*NIM,*sift,*ret_zval;
	XAF_MAT *mat,*newmat;int id;
	KeypointPtr *p;
	newim = NULL;
	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "r", &IM) == FAILURE) {
		return;
	}
	 
	 
	ZEND_FETCH_RESOURCE(im, gdImagePtr, &IM, -1, "Image",le_gd );
	id = test(im,Z_RESVAL_P(IM) TSRMLS_CC);
	p = siftArray();
	ALLOC_INIT_ZVAL(ret_zval);
	array_init(ret_zval);
	while(p){
		ALLOC_INIT_ZVAL(sift); 
		array_init(sift); 
		//php_printf("sssss%g\n",p->descrip);
		add_assoc_double(sift,"descrip", (double)*p->descrip);
		add_assoc_long(sift,"col", p->col);
		add_assoc_long(sift,"row", p->row);
		add_assoc_long(sift,"sx",  p->sx);
		add_assoc_long(sift,"sy", p->sy);
		add_assoc_long(sift,"octave",p->octave);
		add_assoc_long(sift,"ori", p->ori);
		add_assoc_double(sift,"scale", p->scale);
		add_assoc_double(sift,"mag", p->mag);
		add_assoc_long(sift,"level", p->level);
		add_next_index_zval(ret_zval,sift);
		p = p->next;
	}

	RETURN_ZVAL(ret_zval,1,0);
	RETURN_TRUE;
	NIM = createImage( gdImageSX(im)*2, gdImageSY(im)*2 TSRMLS_CC );
	ZEND_FETCH_RESOURCE(newim, gdImagePtr, &NIM, -1, "Image",le_gd );
	mat = createMat(gdImageSY(im),gdImageSX(im),XAF_32FC1);
	xafConvert(im,mat);
	newmat = doubleSizeImage2(mat);
	xafConvert2Image(newmat,newim);
	releaseMat(&mat);
	releaseMat(&newmat);
	
	if(!newim){
		RETURN_FALSE;
	}
	/*/RETURN_FALSE;
	//RETURN_LONG(1);zend_list_insert(im, le_gd TSRMLS_CC)
	//RETURN_RESOURCE(Z_RESVAL_P(IM));*/
	RETURN_RESOURCE(zend_list_insert(newim, le_gd TSRMLS_CC));
	RETURN_ZVAL(NIM,1,0);


	
}

PHP_FUNCTION(xaf_gray)
{
	gdImagePtr im;
	zval *IM;
	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "r", &IM) == FAILURE) {
		return;
	}
	
	ZEND_FETCH_RESOURCE(im, gdImagePtr, &IM, -1, "Image",le_gd );

	grayImage(im);
	RETURN_TRUE;
}

PHP_FUNCTION(xaf_dbimage)
{
	gdImagePtr im;
	gdImagePtr newim;
	zval *IM,*NIM;
	XAF_MAT *mat,*newmat;int id;
	newim = NULL;
	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "r", &IM) == FAILURE) {
		return;
	}
	 
	 
	ZEND_FETCH_RESOURCE(im, gdImagePtr, &IM, -1, "Image",le_gd );
	
	NIM = createImage( gdImageSX(im)*2, gdImageSY(im)*2 TSRMLS_CC );
	ZEND_FETCH_RESOURCE(newim, gdImagePtr, &NIM, -1, "Image",le_gd );
	mat = createMat(gdImageSY(im),gdImageSX(im),XAF_32FC1);
	xafConvert(im,mat);
	newmat = doubleSizeImage2(mat);
	xafConvert2Image(newmat,newim);
	releaseMat(&mat);
	releaseMat(&newmat);
	
	if(!newim){
		RETURN_FALSE;
	}
	RETURN_RESOURCE(zend_list_insert(newim, le_gd TSRMLS_CC));


	

}
/* }}} */
/* The previous line is meant for vim and emacs, so it can correctly fold and 
   unfold functions in source code. See the corresponding marks just before 
   function definition, where the functions purpose is also documented. Please 
   follow this convention for the convenience of others editing your code.
*/


/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * End:
 * vim600: noet sw=4 ts=4 fdm=marker
 * vim<600: noet sw=4 ts=4
 */
