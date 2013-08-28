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

#ifndef PHP_XAF_H
#define PHP_XAF_H

extern zend_module_entry xaf_module_entry;
#define phpext_xaf_ptr &xaf_module_entry

#ifdef PHP_WIN32
#	define PHP_XAF_API __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#	define PHP_XAF_API __attribute__ ((visibility("default")))
#else
#	define PHP_XAF_API
#endif

#ifdef ZTS
#include "TSRM.h"
#endif

#define VALARR_INIT(z,x,y,type) {int next_f = 0; while(1){\
	if(next_f >= y)\
	{\
		break;\
	}\
	z[next_f] = (type*)emalloc(sizeof(type)*x);\
	next_f++;\
}}

#define VALARR_FREE(z,num) {int next_f = 0; while(1){\
	if(next_f >= num)\
	{\
		break;\
	}\
	efree(z[next_f]);\
	next_f++;\
}}

#define ARRVAL_INDEX(ht,index,data,pointer) {int next_f = 0; while(1){\
	if(next_f == index){ \
		zend_hash_get_current_data_ex(ht,(void**)data,pointer);\
		break;\
	}\
	zend_hash_move_forward_ex(ht,pointer);\
	next_f++;\
}}
#define ZVALARR_INIT(z,num) {int next_f = 0; while(1){\
	if(next_f >= num)\
	{\
		break;\
	}\
	MAKE_STD_ZVAL(z[next_f])\
	ZVAL_NULL(z[next_f]);\
	next_f++;\
}}

#define ZVALARR_FREE(z,num) {int next_f = 0; while(1){\
	if(next_f >= num)\
	{\
		break;\
	}\
	efree(z[next_f]);\
	next_f++;\
}}

#ifndef PI
#define PI 3.14159265358979323846
#endif

#define CHARBIT 3
#define BITMASK(b) (1<<(b - (b>>CHARBIT<<CHARBIT)))
#define BITSLOT(b) (b>>CHARBIT)
#define BITSET(a, b) ((a)[BITSLOT(b)] |= BITMASK(b))
#define BITCLEAR(a, b) ((a)[BITSLOT(b)] &= ~BITMASK(b))
#define BITTEST(a, b) ((a)[BITSLOT(b)] & BITMASK(b))
#define BITNSLOTS(nb) ((nb+7)>>CHARBIT)

#define cscGr_32f  0.299f
#define cscGg_32f  0.587f
#define cscGb_32f  0.114f 
/*
ZEND_BEGIN_MODULE_GLOBALS(xaf)
	int le_gd;
ZEND_END_MODULE_GLOBALS(xaf)
*/
/*ZEND_DECLARE_MODULE_GLOBALS(xaf);*/
PHP_MINIT_FUNCTION(xaf);
PHP_MSHUTDOWN_FUNCTION(xaf);
PHP_RINIT_FUNCTION(xaf);
PHP_RSHUTDOWN_FUNCTION(xaf);
PHP_MINFO_FUNCTION(xaf);

PHP_FUNCTION(confirm_xaf_compiled);	/* For testing, remove later. */
PHP_FUNCTION(xaf_rtype);
PHP_FUNCTION(xaf_imgDct);
PHP_FUNCTION(xaf_hmdist);
PHP_FUNCTION(xaf_sift);
PHP_FUNCTION(xaf_gray);

long  colorPoint(gdImagePtr im,int x,int y);
int colorChange(gdImagePtr im,int color,int x,int y);
void grayImage(gdImagePtr im);
static double average(double** pix, int w, int h);
/* 
  	Declare any global variables you may need between the BEGIN
	and END macros here:     
*/



/* In every utility function you add that needs to use variables 
   in php_xaf_globals, call TSRMLS_FETCH(); after declaring other 
   variables used by that function, or better yet, pass in TSRMLS_CC
   after the last function argument and declare your utility function
   with TSRMLS_DC after the last declared argument.  Always refer to
   the globals in your function as XAF_G(variable).  You are 
   encouraged to rename these macros something shorter, see
   examples in any other php module directory.
*/

#ifdef ZTS
#define XAF_G(v) TSRMG(xaf_globals_id, zend_xaf_globals *, v)
#else
#define XAF_G(v) (xaf_globals.v)
#endif

#endif	/* PHP_XAF_H */


/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * End:
 * vim600: noet sw=4 ts=4 fdm=marker
 * vim<600: noet sw=4 ts=4
 */
