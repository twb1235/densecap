from django.shortcuts import render, HttpResponse, redirect
from imgapp.models import IMG
from imgapp.algrithm_pkg import describe
#from imgapp.ch_image_cap import generate_caption
import os
#import tensorflow as tf
#graph = tf.get_default_graph()
import base64
# Create your views here.
def show(request):
    if request.method == 'GET':
        content = {
            'img_url': '',
            'caption': '',
        }
        return render(request, 'imgapp/upload2show.html', content)
    if request.method == 'POST':
        new_img = IMG(
            # img=request.FILES.get('img')
            img=request.FILES['img']
        )
        new_img.save()
        img_url = IMG.objects.all()[::-1][0].img.url
        
        #describe.caption(img_url='imgcap_django'+img_url)
        #caption='imgapp/algrithm_pkg/result/123.jpg'
        #caption 图片路径
        #if language == 1 or language == 0:
         #   global graph
          #  with graph.as_default():
           #     caption = generate_caption.generate_caption('/home/anna/pycharm_proj/imgcap_django'+img_url)
        #elif language == 2:
         #   caption = imgcap2django.cal_caption(img_url='/home/anna/pycharm_proj/imgcap_django'+img_url, beam_size=5)
        #else:
        #将此图片进行
        s=describe.caption(img_url)

        caption = 'media/result/'+s
        content = {
            'text1': 'This is your submit:',
            'text2': 'This is a caption for your image:',
            'img_url': img_url,
            'caption': caption
        }
        agent = ''.join(request.META.get('HTTP_USER_AGENT'))
        # print(agent)
        if 'Android' in str(agent):
            print('There is a Android')
            with open(caption, 'rb') as f:
                image_data = f.read()
            return HttpResponse(base64.b64encode(image_data), content_type="image/jpg")
        elif 'iPhone' in str(agent):
            print('There is a iPhone')
            with open(caption, 'rb') as f:
                image_data = f.read()
            return HttpResponse(base64.b64encode(image_data), content_type="image/jpg")
        elif 'Linux' in str(agent):
            print('There is a Linux')
            return render(request, 'imgapp/upload2show.html', content)
        elif 'Windows' in str(agent):
            print('There is a Windows')
            return render(request, 'imgapp/upload2show.html', content)

# def show(request):
#     if request.method == 'POST':
#         new_img = IMG(
#             # img=request.FILES.get('img')
#             img=request.FILES['img']
#         )
#         new_img.save()
#
#     img_url = IMG.objects.all()[::-1][0].img.url
#     # img = IMG.objects.get(name='img')
#     # if not img.exists():
#     #     img = IMG(img='default.jpg')
#     caption = imgcap2django.cal_caption(img_url='/home/anna/pycharm_proj/imgcap_django'+img_url, beam_size=5)
#     content = {
#         'img_url': img_url,
#         'caption': caption,
#     }
#     return render(request, 'imgapp/upload2show.html', content)
