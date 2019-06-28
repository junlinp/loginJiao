#!/usr/bin/python
#coding:utf-8
import os
import urllib
import json
import requests
import cv2
import numpy as np
import time
import model

def getCookie():
    resp = urllib.request.urlopen("http://210.42.121.241/servlet/GenImg")

    with open(str(os.getpid()) + ".jpeg", 'wb') as f:
        f.write(resp.read())

    return resp.info()['set-cookie']

def checkCode(Indentify_code, cookie):
    url = "http://210.42.121.241/servlet/Login"

    data = {'id' : "", 'pwd' : "", "xdvfb" : Indentify_code}
    data = 'id=&pwd=&xdvfb=' + str(Indentify_code)

    head = {'Cookie' : cookie,
    'Connection' : 'keep-alive',
    'Content-Type': "application/x-www-form-urlencoded",
    "User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:50.0) Gecko/20100101 Firefox/50.0",
    "Accept-Encoding": "gzip, deflate",
    "Referer" :"http://210.42.121.132/",
    "Host" : "21042.121.132",
    "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Upgrade-Insecure_Requests" : "1"
    }

    r = requests.post(url, data = data, headers = head, allow_redirects=False)
    print (r.status_code)
    assert r.status_code == 302 or r.status_code == 200
    if r.status_code == 302:
        return True
    else:
        return False



network = model.Model()
network.build_model()
network.load_model("model.m")
def getDate():
    cookie = getCookie()


    img = cv2.imread(str(os.getpid()) + ".jpeg")
    img = [img]
    img = np.asarray(img) / 255.0
    predict = network.predict(img)

    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C', 'D','E','F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    indentify_code = label[predict[0]] + label[predict[1]] + label[predict[2]] + label[predict[3]]
    if not os.path.exists("prediction"):
        os.mkdir("prediction")
    img = cv2.imread(str(os.getpid()) + ".jpeg")
    cv2.imwrite(os.path.join("prediction", indentify_code + ".jpg"), img)
    print( "IndentifyCode : {}".format(indentify_code))
    result =  checkCode(indentify_code, cookie)
    #print result
    if result == True:
        Path = './Valid'
        if not os.path.exists(Path):
            os.mkdir(Path)
        img = cv2.imread(str(os.getpid()) + ".jpeg")
        cv2.imwrite("./t/" + indentify_code + ".jpg", img)
        return True
    return False

def process():
    count = 1
    True_count = 0
    for i in range(100000):
        result = getDate()
        if result is True:
            True_count = True_count +  1
        count += 1
        print ("%s : accuracy: %f" % (os.getpid(), True_count * 100.0 / count) )


if __name__ == '__main__':
    process()
    #print(getCookie())
    #print( checkCode("MD43", "JSESSIONID=ABE00A5E161D54062AD4F488340AC324; Path=/") )



