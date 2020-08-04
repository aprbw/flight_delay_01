# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:02:19 2018

@author: Arian Prabowo
"""
import time
import multiprocessing as mp

class ProgressIndicator:
    '''
    n=12345
    pi = ProgressIndicator(n)
    for i in range(n):
        time.sleep(0.001)
        pi.toc()
    '''
    defaultUpdatePeriod = 1
    def __init__(self,n = None):
        self.reinit(n)
    def reinit(self,n=None):
        self.timeStart = time.perf_counter()
        self.updatePeriod = ProgressIndicator.defaultUpdatePeriod
        self.lastToc = time.perf_counter()
        self.lock = mp.Lock()
        self.i = mp.Value('i',1)
        if n != None:
            if n <= 0:
                n = 1
            self.n=n
            self.nstr = '{:,}'.format(self.n) # n in string
            self.nstrw = len(self.nstr) # width of n in string
            self.toc = self.toc_with_n
        else:
            self.toc = self.toc_without_n
    def tic(self):
        self.timeStart = time.perf_counter()
        with self.lock:
            self.i.value=1
        self.lastToc=time.perf_counter()
    def toc_with_n(self,str_remarks='',ai=None):
        if ai != None:
            if ai <= 0:
                ai = 1
            with self.lock:
                self.i.value=ai
        td = time.perf_counter()-self.lastToc
        str_return = ''
        if td >= self.updatePeriod:
            te=time.perf_counter()-self.timeStart # time elapsed
            he, re = divmod(te, 3600) # hour and remainder elapsed
            me, se = divmod(re, 60)   # minutes and second elapsed
            with self.lock:
                tt=te*self.n/self.i.value       # time total
                ht, rt = divmod(tt, 3600) # hour and remainder total
                mt, st = divmod(rt, 60)   # minute and second total
                str_return = '\r'+'{:.2%}'.format(self.i.value/self.n)+' | ' + \
                      '{:0>{width},}'.format(self.i.value,width=self.nstrw)+'/'+self.nstr+' | '+ \
                      '{:0>2,.0f}:{:0>2,.0f}:{:0>2,.0f}'.format(he,me,se)+'/'+ \
                      '{:0>2,.0f}:{:0>2,.0f}:{:0>2,.0f}'.format(ht,mt,st)+' | '+ \
                      str_remarks
            print(str_return,end='',flush=True)
            self.lastToc=time.perf_counter()
        with self.lock:
            self.i.value+=1
        return str_return
    def toc_without_n(self,str_remarks=''):
        td = time.perf_counter()-self.lastToc
        str_return = ''
        if td >= self.updatePeriod:
            te=time.perf_counter()-self.timeStart # time elapsed
            he, re = divmod(te, 3600) # hour and remainder elapsed
            me, se = divmod(re, 60)   # minutes and second elapsed
            with self.lock:
                str_return = '\r'+ \
                      str(self.i.value)+' | '+ \
                      '{:0>2,.0f}:{:0>2,.0f}:{:0>2,.0f}'.format(he,me,se)+' | '+ \
                      str_remarks
            print(str_return,end='',flush=True)
            self.lastToc=time.perf_counter()
        with self.lock:
            self.i.value+=1
        return str_return