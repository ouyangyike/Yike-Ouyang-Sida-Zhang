import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw,ImageFont

'''
set up training_set, training_labels and so on.
Like initializtion
'''
SamplePaths=["D:\\svm\\imagecat\\","D:\\svm\\imagemaltese\\","D:\\svm\\imagefox\\","D:\\svm\\imagepanda\\","D:\\svm\\imagetiger\\"]
SampleLabels=[1,2,3,4,5]
accuracyresult=[]
samplenumber=800
for k_num in range(1,31):
    SvmList=[]
    print("FirstK:"+str(k_num))
    for sampleindex1 in xrange(4):
        for sampleindex2 in range(sampleindex1+1,5):
            tempsamplepath1=SamplePaths[sampleindex1]
            tempsamplelabel=SampleLabels[sampleindex1]
            tempsamplepath2=SamplePaths[sampleindex2]
            tempsamplelabe2=SampleLabels[sampleindex2]
            print(samplenumber)
            training_set=[]
            training_labels=[]
            svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                                svm_type = cv2.SVM_C_SVC,
                                C=2.67, gamma=5.383 )
            

            '''
            produce the traning data set of cat and extract key points by SIFT and kmeans
            '''

            for i in xrange(samplenumber):
                try:
                    img=cv2.imread(tempsamplepath1+str(i)+".jpg")
                    res=cv2.resize(img,(250,250))
                    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    sift = cv2.SIFT()
                    kp, des = sift.detectAndCompute(gray,None)
                    temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
                    meanstest=meanstest.reshape(-1)
                    training_set.append(meanstest)
                    training_labels.append(tempsamplelabel)
                    trainData=np.float32(training_set)
                    responses=np.float32(training_labels)
                except BaseException as error:
                    print("1 error")
                    continue
                #print("running")
                
            '''
            produce the traning data set of maltese and extract key points by SIFT and kmeans
            '''

            for i in xrange(samplenumber):
                try:
                    img=cv2.imread(tempsamplepath2+str(i)+".jpg")
                    res=cv2.resize(img,(250,250))
                    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    sift = cv2.SIFT()
                    kp, des = sift.detectAndCompute(gray,None)
                    temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
                    meanstest=meanstest.reshape(-1)
                    training_set.append(meanstest)
                    training_labels.append(tempsamplelabe2)
                    trainData=np.float32(training_set)
                    responses=np.float32(training_labels)
                except BaseException as error:
                    print("2 error")
                    continue
                #print("running")
                
            
            '''
            until this part, the training data has been created and next we will use it to train different decision boundarys through SVM
            '''
            arr= np.array(training_labels)
            print(arr.shape)
            print(trainData.shape)
            training_labels=arr
            print(trainData.shape)
            print(training_labels)
            responses=np.float32(training_labels)
            svm = cv2.SVM()
            svm.train(trainData,responses, params=svm_params)
            SvmList.append(svm)
    print("secondK:"+str(k_num))
    Paths=[]
    training_set=[]
    training_labels=[]

    for i in xrange(10):
        try:
            img=cv2.imread("D:\\svm\\imagecat\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(1)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            '''
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("D:\\svm\\imagecat\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:1",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="D:\\svm\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
            '''
        except BaseException as error:
            print("error")
            continue
    for i in xrange(10):
        try:
            img=cv2.imread("D:\\svm\\imagemaltese\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(2)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            '''
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("D:\\svm\\imagemaltese\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:2",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="D:\\svm\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
            '''
        except BaseException as error:
            print("error")
            continue
    for i in xrange(10):
        try:
            img=cv2.imread("D:\\svm\\imagefox\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(3)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            '''
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("D:\\svm\\imagefox\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:3",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="D:\\svm\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
            '''
        except BaseException as error:
            print("error")
            continue
    for i in xrange(10):
        try:
            img=cv2.imread("D:\\svm\\imagepanda\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(4)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            '''
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("D:\\svm\\imagepanda\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:4",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="D:\\svm\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
            '''
        except BaseException as error:
            print("error")
            continue
    for i in xrange(10):
        try:
            img=cv2.imread("D:\\svm\\imagetiger\\"+str(i)+".jpg")
            res=cv2.resize(img,(250,250))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            temptest, classified_pointstest, meanstest=cv2.kmeans(data=des,K=k_num,bestLabels=None,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),attempts=1,flags=cv2.KMEANS_RANDOM_CENTERS)
            meanstest=meanstest.reshape(-1)
            training_set.append(meanstest)
            training_labels.append(5)
            trainData=np.float32(training_set)
            responses=np.float32(training_labels)
            '''
            font=cv2.FONT_HERSHEY_SIMPLEX
            im = Image.open("D:\\svm\\imagetiger\\"+str(i)+".jpg").convert('RGBA')
            im=im.resize((250,250))
            txt=Image.new('RGBA', im.size, (0,0,0,0))
            fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
            d=ImageDraw.Draw(txt)
            d.text((txt.size[0]-240,txt.size[1]-60), "true label:5",font=fnt, fill=(0,255,0,255))
            #d.text((txt.size[0]-120,txt.size[1]-60), "test label:1",font=fnt, fill=(255,0,0,255))
            out=Image.alpha_composite(im, txt)
            #out.show()
            temppath="D:\\svm\\testset\\"+"Knum_"+str(k_num)+"i_"+str(trainData.shape[0])+".jpg"
            out.save(temppath)
            Paths.append(temppath)
            '''
        except BaseException as error:
            continue
    result=[]
    print("K:"+str(k_num))
    print("snum:"+str(samplenumber))
    for j in xrange(len(trainData)):
        countresult=[0,0,0,0,0]
        for i in xrange(len(SvmList)):
            svm=SvmList[i]
            countresult[int(svm.predict(trainData[j])-1)]+=1
        result.append(countresult.index(max(countresult))+1)
    print(training_labels)
    print(result)
    rightnum=0
    for h in xrange(len(result)):
        '''
        temppath=Paths[h]
        im = Image.open(temppath).convert('RGBA')
        im=im.resize((250,250))
        txt=Image.new('RGBA', im.size, (0,0,0,0))
        fnt=ImageFont.truetype("c:/Windows/fonts/Tahoma.ttf", 20)
        d=ImageDraw.Draw(txt)
        d.text((txt.size[0]-120,txt.size[1]-60), "test label:"+str(int(result[h])),font=fnt, fill=(255,0,0,255))
        out=Image.alpha_composite(im, txt)
        #out.show()
        out.save(temppath)
        '''
        if training_labels[h]==result[h]:
            rightnum+=1
    tempaccuracy=rightnum/(len(result)*1.0)
    accuracyresult.append(tempaccuracy) 
print(accuracyresult)
accuracyresult=np.array(accuracyresult)
plt.figure(0)
plt.clf()
accuracyresultnum = np.array(accuracyresult)
kindex=[]
for index in xrange(len(accuracyresultnum)):
    kindex.append(index)
kindex=np.array(kindex)
plt.plot(kindex[:], accuracyresultnum[:], 'b', label='Test')
plt.xlabel('Samples')
plt.ylabel("Accuracy")
plt.ylim(0.0,1.1)
plt.legend()
plt.draw()
plt.pause(0.0001)
