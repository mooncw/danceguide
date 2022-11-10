from django.shortcuts import render

from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading

from pathlib import Path

from dance.models import Dancemusic

from .mpi import net, mpi_keypoint

def home(request):
    music_list = Dancemusic.objects.all()
    return render(request, 'home.html', {'music_list': music_list})

def pose(request):
    return render(request, 'pose.html')

def video(request, music):
    music_query = Dancemusic.objects.get(pk = music)
    # music_name = Dancemusic.objects.get(pk = music).music_name
    # music_id = Dancemusic.objects.get(pk = music).music_id
    # music_path = Dancemusic.objects.get(pk = music).music_path
    return render(request, 'video.html', {'music': music, 'music_query': music_query})
    # return render(request, 'video.html', {'music_id': music_id, 'music_path': music_path, 'music_name': music_name})

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        image = mpi_keypoint(image, net)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def detectme(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass