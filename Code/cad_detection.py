# -*- coding: utf-8 -*-
from PIL import ImageGrab, Image, ImageOps
import os
# import glob
# import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import model_from_json
import cv2
from tensorflow.keras.models import load_model
from skimage import color
from sklearn import preprocessing


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit_transform(['decimal','div' ,'eight' ,'equal' ,'five' ,'four' ,'minus' ,'nine' ,'one' ,'plus' ,'seven' ,'six' ,'three' ,'times' ,'two' ,'zero'])

labels = ['decimal', 'div', 'eight', 'equal', 'five', 'four', 'minus', 'nine', 'one', 'plus', 'seven', 'six', 'three',
          'times', 'two', 'zero']

# 모델 저장 파일 위치
output_folder_name = "..\\Output"
# 테스트 이미지
test_image_path = "..\\TestImages\\mywrite.png"
# 학습 이미지 크기
dims = (100, 100)
# 이미지 체널 수
n_channels = 1
# 윈도우 이동 간격
step_size = 20  # 16
# 윈도우 크기    
window_size = dims
# 최소 이미지 크기
min_size = (100, 100)  # 100, 100
# 이미지 축소 비율
downscale = 1.4
# NMS threshold
threshold = 0.000001

'''
#%% 디버깅:폴더에서 임의의 이미지를 읽는다.
def load_test_images():
    # 자동차 이미지를 저장하기 위한 배열
    image_filename_cars = []
    # 비-자동차 이미지를 저장하기 위한 배열
    image_filename_notcars = []

    # DataSet 폴더에 있는 모든 png 파일명을 가져온다.
    image_filenames = glob.glob('../DataSet/*/*/*.png')
    
    # 각각의 파일명에 대해...
    for image_filename in image_filenames:
        # 경로에 'non-vehicle'가 있으면 비-자동차로 인식한다.
        if 'non-vehicle' in image_filename:
            # 비-자동차 파일명을 image_filename_notcars에 추가한다.
            image_filename_notcars.append(image_filename)
        else:
            # 자동차 파일명을 image_filename_notcars에 추가한다.
            image_filename_cars.append(image_filename)
            
    # 파일을 임의로 한 개 골라 이미지를 읽는다.
    image_car = mpimg.imread(random.choice(image_filename_cars))
    image_notcar = mpimg.imread(random.choice(image_filename_notcars))
        
    # 이미지를 리턴한다.
    return image_car, image_notcar

#%% 디버깅: 모델을 이용해 예측한 결과를 보여준다.
def show_prediction(model, image):
    # 화면에 이미지를 보여준다.
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    # 결과를 예측한다.
    predicted = model.predict(image.reshape(1, *dims, n_channels))
    # 예측한 결과를 출력한다.
    if predicted >= 0.5:
        print("Predicted = Car")
    else:
        print("Predicted = Not a Car")    
    
    return
'''


# %% 축소한 이미지를 yield를 이용해 반복자로 반환한다.
def pyramid(image, downscale=1.5, min_size=(64, 64)):
    # 원래 이미지를 yield한다.
    yield image

    # 단계적으로 이미지를 축소하고, yield한다.
    while True:
        # 축소할 이미지 크기를 계산한다.
        w = int(image.shape[1] / downscale)
        # 이미지 크기를 줄인다.
        image = resize(image, width=w)

        # 축소한 이미지 크기가 min_size보다 작으면 멈춘다.
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

            # 축소한 이미를 yield한다.
        yield image

    # %% 이미지 크기를 줄인다.


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 이미지 크기
    (h, w) = image.shape[:2]

    # width와 height가 모두 None이면 원본 이미지를 반환한다.
    if width is None and height is None:
        return image

        # width가 None이면...
    if width is None:
        # height에 맞게 width를 계산한다.
        r = height / float(h)
        dim = (int(w * r), height)
        # height가 None이면...
    else:
        # width에 맞게 height를 계산한다.
        r = width / float(w)
        dim = (width, int(h * r))

        # OpenCV를 이용해 이미지 크기를 조정한다.
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


# %% 슬라이딩 윈도우를 yield를 이용해 반복자로 반환한다.
def sliding_window(image, step_size, window_size):
    # image 전체에 대해 가로, 세로 방향으로 일정한 step_size 만큼 이동한다.
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # 현재 윈도우 위치(x,y)에서 window_size만큼의 부분 이미지를 반환한다.
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# %% NMS 구현
def non_max_suppression(boxes, class_scores, overlap_thresh=0.7):
    # 박스가 없으면 종료
    if len(boxes) == 0:
        return []

    # boxes가 정수면 float로 변환. 나중에 나눗셈 연산을 정확하게 하기 위해
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # boxes들의 각 좌표값
    # TODO: add process for class index 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 각 box 면적을 계산
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # box를 정렬한다.
    idxs = np.argsort(y2)

    # 각각의 box에 대해서...
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 겹치는 부분을 계산한다.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 겹치는 부분의 크기를 계산
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 겹치는 부분의 면적 비율을 계산
        overlap = (w * h) / area[idxs[:last]]

        # 겹치는 부분이 overlap_thresh 이상이면 지운다. 즉, 다른 박스와 겹치기 때문에 없어도 된다.
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # 남아있는 박스를 반환한다.
    return boxes[pick].astype("int"), class_scores[pick]


def is_all_black(img):
    normed = img / 255
    mean = normed.mean()
    if np.abs(mean - 1.0) <= 0.0000001:
        return True
    else:
        return False


# grey scale image with white background
def get_centroid(image):
    image = 255 - image
    m = cv2.moments(image)
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return x, y


def is_centroid_in_window(x, y, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x1 < x < x2) and (y1 < y < y2)


# %% 메인 함수
def main():
    # # 모델을 불러온다.
    # with open(os.path.join(output_folder_name, 'model_architectures.json'), 'r') as f:
    #     model = model_from_json(f.read())
    #
    # # 가중치를 불러온다.
    # model.load_weights(os.path.join(output_folder_name, 'model_weights.h5'))
    # # 모델을 컴파일한다.
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model = load_model(os.path.join(output_folder_name, 'mix.h5'))


    # 디버깅용: 학습 모델을 테스트 한다.
    # 모델을 표시한다.
    # model.summary()
    # 모델 테스트를 위해서 Dataset 폴더에서 자동차 이미지와 비-자동차 이미지를 임의로 한 개 읽어온다.
    # image_car, image_notcar = load_test_images()
    # image_notcar = mpimg.imread('..\\test.png')
    # image_notcar.reshape(1, *dims, n_channels)
    # 불러온 image_car 이미지를 이용해 예측한다.
    # print("Actual = Car")
    # show_prediction(model, image_car)
    # print("Actual = Not a Car")
    # show_prediction(model, image_notcar)

    # 테스트용 이미지를 불러온다.
    # test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    img = Image.open(test_image_path)
    img = img.convert('L')
    # img = ImageOps.invert(img)
    pix = np.array(img)
    # 분류기가 0~1 스케일을 사용하였기 때문에 이미지 스케일을 0~255에서 0~1로 조정한다.
    # test_image = pix.astype(np.float32) / 255
    test_image = pix.astype(np.float32) / 1

    # 이미지 처리 후 저장할 파일 이름
    path, filename = os.path.split(test_image_path)
    filename = os.path.splitext(filename)[0]
    test_image_before_nms_path = os.path.join(path, filename + '_before_nms.png')
    test_image_after_nms_path = os.path.join(path, filename + '_after_nms.png')

    # 화면에 이미지를 보여준다.    
    plt.imshow(test_image)
    plt.title('Original image')
    plt.xticks([]), plt.yticks([])
    plt.show()

    # 검색 결과를 저장할 리스트. 박스의 양 끝 좌표(x1,y1,x2,y2)가 저장됨
    detections = []
    detections_c = []
    # downscale 조정 인자
    downscale_power = 0
    # 이미지 복사본
    test_image_clone = test_image.copy()
    # 이미지 피라미드를 이용해 이미지를 단계적으로 축소시킨다.
    for scaled_image in pyramid(test_image, downscale, min_size):
        # 슬라이딩 윈도우 내의 부분 이미지에 적용한다.
        for (x, y, window) in sliding_window(scaled_image, step_size, window_size):
            # 크기가 맞지 않는다면 무시한다. 이미지는 (세로 크기, 가로 크기)로 저장됨
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
            # print("window {}, {} at {} {} on scaled image size{}".format(window_size[0], window_size[1], x, y, scaled_image.shape))
            # 분류기를 이용해 자동치인지 예측한다.
            # reshaped = window.reshape(1, *dims, n_channels)
            if is_all_black(window):
                continue
            wind = np.expand_dims(window, axis=0)
            reshaped = wind.reshape(*dims, 1)
            expanded = np.expand_dims(reshaped, axis=0)
            predicted = model.predict(expanded)
            # 자동차라고 예측했다면...
            # predicted = predicted.reshape(-1)

            def ret_to_label(pred):
                arg_sorted = np.argsort(pred)
                max_idx = arg_sorted[0][-1]
                final_label = None
                score = 0.0
                if pred[0][max_idx] > 0.5:
                    final_label, score = labels[max_idx], pred[0][max_idx]
                return final_label, score

            label, score = ret_to_label(predicted)
            if label is not None:
                # 축소 전의 이미지 상 위치를 계산한다.
                x1 = int(x * (downscale ** downscale_power))
                y1 = int(y * (downscale ** downscale_power))
                # 윈도우 네 모서리 좌표를 detections에 저장한다.
                detections.append((x1, y1,
                                   x1 + int(window_size[0] * (downscale ** downscale_power)),
                                   y1 + int(window_size[1] * (downscale ** downscale_power))))
                detections_c.append((label, score))

        # downscale 조정 인자를 조절한다.
        downscale_power += 1

        # 찾은 윈도우들을 보여준다.
    test_image_before_nms = test_image_clone.copy()
    for (x1, y1, x2, y2) in detections:  # TODO: process class label
        # 이미지에 윈도우를 그린다.
        cv2.rectangle(test_image_before_nms, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)

        # 화면에 이미지를 표시한다.
    plt.title('Detected cars befor NMS')
    plt.imshow(test_image_before_nms)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imsave(test_image_before_nms_path, test_image_before_nms)

    #
    def math_symbol_suppression(boxes, classes_score, overlap_thresh=0.7):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("int")

        pick, ret = [], []
        prev_c = ""
        prev_i = -1
        prev_s = 0.0
        # reduce and only leave nearby max score
        for i in range(len(classes_score)):
            c, s = classes_score[i]
            if c != prev_c:  # anyway need record
                if prev_i == -1:  # first
                    prev_c, prev_s, prev_i = c, s, i
                else:  # new class starts, append old class highest
                    pick.append(prev_i)
                    prev_c, prev_s, prev_i = c, s, i
            else:  # the same class continues
                if prev_s < s:  # prev score is smaller, use current
                    prev_c, prev_s, prev_i = c, s, i
                else:
                    pass  # still use old
        pick.append(prev_i)

        # reduce overlap by centroid
        def is_in_ret_boxes(ret_idx, boxes, centroid, c_s):
            if not ret_idx:
               return False
            x, y = centroid
            for i in ret_idx:
                in_box = is_centroid_in_window(x, y, boxes[i])
                if in_box:
                    return True
                else:
                    continue
            return False

        for i in pick:
            box = boxes[i]
            x1, y1, x2, y2 = box
            box_image = test_image_clone[y1: y2, x1: x2]
            center = get_centroid(box_image)
            x = center[0] + x1
            y = center[1] + y1
            in_ret = is_in_ret_boxes(ret, boxes, (x, y), classes_score[i])
            if in_ret:
                continue  # abandon it as it overlaps with previous one
            else:
                ret.append(i)

        return boxes[ret].astype("int"), classes_score[ret]

    # Non-Maxima Suppression을 수행한다.
    detections_nms, detections_c_nms = math_symbol_suppression(np.array(detections), np.array(detections_c), threshold)
    print("detections after nms " + str(detections_c_nms))
    test_image_after_nms = test_image_clone
    for (x1, y1, x2, y2) in detections_nms:
        # 이미지에 윈도우를 그린다.
        cv2.rectangle(test_image_after_nms, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)

        # 화면에 이미지를 표시한다.
    plt.title('Detected cars after NMS')
    plt.imshow(test_image_after_nms)
    plt.xticks([]), plt.yticks([])
    plt.show()
    plt.imsave(test_image_after_nms_path, test_image_after_nms)
    pass
    return


if __name__ == "__main__":
    main()
