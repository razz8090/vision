from django.shortcuts import render

from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.conf.urls.static import static
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import os
import cv2
import keras
import keras.backend.tensorflow_backend as tb
from vison_app import models



@csrf_exempt
def homePage(request):
    return render(request,'index.html')



@csrf_exempt
def pic_upload(request):
    try:
        file=request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        upload_url=fs.url(filename)
        path = "Vi_project/media/"+file.name
        output= models.read_photo(path)
        if output != "No_face" and output != "Multiple_face":
            try:
                gender,age = models.predition_age_gender(output)
                return render(request,'page2.html',{'pic_name':upload_url,'age':age,'gender':gender})
            except Exception as error:
                return render(request,'page2.html',{'error':str(error),'pic_name':upload_url})
        else:
            return render(request,'page2.html',{'error':output,'pic_name':upload_url})

    except:
        return HttpResponseRedirect("/")