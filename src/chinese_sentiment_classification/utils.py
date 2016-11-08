#!/usr/bin/env pyton
#coding=utf-8
import sys
import xlrd
def excel2txt(excel_path='',txt_path=''):
    bk = xlrd.open_workbook(excel_path)
    sheetnames = bk.sheet_names()
    for m in sheetnames:
        sh = bk.sheet_by_name(m)
        nrows = sh.nrows
        f = open(txt_path,'a')
        for i in range(nrows):
            row_data = sh.row_values(i)
            for j in range(len(row_data)):
                s = row_data[j].encode('utf-8')
                f.write(s+'\n')
    f.close()
#fname = u'F:/迅雷下载/chrome download/pos.xls'
#txt_path = u'./pos.txt'
#excel2txt(fname,txt_path)