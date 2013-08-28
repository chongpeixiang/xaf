dnl
dnl $Id$
dnl

PHP_ARG_ENABLE(xaf, whether to enable sift support,
[  --disable-xaf          Disable sift support], yes)

if test "$PHP_XAF" != "no"; then
  AC_DEFINE([HAVE_XAF],1 ,[whether to enable sift support])
  AC_HEADER_STDC

  PHP_NEW_EXTENSION(xaf, xaf.c sift.c, $ext_shared)
  PHP_INSTALL_HEADERS([ext/xaf], [php_xaf.h])
  PHP_SUBST(XAF_SHARED_LIBADD)
fi
