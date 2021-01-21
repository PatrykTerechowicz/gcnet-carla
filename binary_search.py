# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:38:56 2021

@author: Lab_admin
"""

def search_frame(seq, frame_id):
    '''Binary search'''
    l = 0
    r = len(seq)-1
    while not l == r:
        d = (l+r)//2
        el = seq[d][1]
        if el == frame_id:
            return d
        elif el > frame_id:
            r = d-1
        elif el < frame_id:
            l = d+1
    return -1


def test_search_empty():
    seq = [(0, i) for i in range(10)]
    x = search_frame(seq, 10)
    print(x)


def test_search():
    # search_frame func test
    c_t = time()
    seq = [(0, i*2) for i in range(1000)]
    n_p = list(range(1, 1000, 2))
    p_p = list(range(0, 1000, 2))
    pass_n = 0
    test_n0 = 2
    #test theese el which are in seq
    for i in range(test_n0):
        searched = choice(p_p)
        found = search_frame(seq, searched)
        print(searched)
        print(found)
        if seq[found][1] == searched:
            pass_n += 1
    for i in range(test_n0):
        searched = choice(n_p)
        found = search_frame(seq, searched)
        if found == -1:
            pass_n += 1
    print("Passed %d tests out of %d" % (pass_n, 2*test_n0))
    e_time = time() - c_t
    print("This took %.2fs" % e_time)