import os
import numpy as np
from PIL import Image, ImageDraw, ImageWin, ImageTk
import matplotlib.pyplot as plt
from ipywidgets.widgets import widget

import cv2 as cv
import operator

from config import SBU_test_pth, BDRAR_SBU_Detection, Direction_SBU_Detection, Distraction_SBU_Detection
from misc import check_mkdir

from deep_dist import DeepRank


def compute_IOU(im_pred, im_lab):
    im_pred = np.asarray(im_pred).copy()
    im_lab = np.asarray(im_lab).copy()

    overlap = np.sum((im_pred > 0) * (im_lab > 0))
    union = np.sum((im_pred + im_lab) > 0)

    iou = overlap / union

    return iou


def compute_BER(im_pred, im_lab):
    im_pred = np.asarray(im_pred).copy()
    im_lab = np.asarray(im_lab).copy()

    imPred_p = im_pred > 0
    imPred_n = im_pred == 0
    imLab_p = im_lab > 0
    imLab_n = im_lab == 0

    N_tp = np.sum(imPred_p * imLab_p)
    N_tn = np.sum(imPred_n * imLab_n)
    N_p = np.sum(imLab_p)
    N_n = np.sum(imLab_n)

    BER = 1 - 0.5 * (N_tp / N_p + N_tn / N_n)

    return BER


def compute_AP():
    return 0


def show_image(name_dict):

    for key in name_dict:

        image = cv.imread(name_dict[key])
        cv.imshow(str(key), image)

    cv.waitKey(0)


to_test = {'SBU_test_pth': SBU_test_pth}
to_predictions = {'Direc_SBU_Detection': Direction_SBU_Detection,
                  'BDRAR_SBU_Detection': BDRAR_SBU_Detection,
                  'Distr_SBU_Detection': Distraction_SBU_Detection}

model_path = '/home/wilson/DE/Python/Image-Similarity-Deep-Ranking-master/Model/deepranking-v2-150000.h5'

if __name__ == '__main__':

    BER = []

    deep_rank = DeepRank(model_path)

    img_number = 0

    predictions = {}

    method_rank = {}

    for name, label_root in to_test.items():

        img_list = [img_name for img_name in os.listdir(os.path.join(label_root, 'ShadowMasks')) if
                    img_name.endswith('png')]

        img_number = len(img_list)

        for idx, img_name in enumerate(img_list):
            print('evaluation for %s: %d / %d' % (name, idx + 1, len(img_list)))
            # check_mkdir()

            label_name = os.path.join(label_root, 'ShadowMasks', img_name)
            label_img = Image.open(label_name)

            ori_name = os.path.join(label_root, 'ShadowImages', img_name[:-4] + '.jpg')
            ori_img = cv.imread(ori_name)

            # cv.imshow("Original Image", ori_img)

            # cv.imshow("label", cv.cvtColor(np.asarray(label_img), cv.COLOR_RGB2BGR))

            ber_mul_best = []
            iou_mul = {}
            ber_mul = {}

            img_set = {}

            img_set.update({'ori_image': ori_name})
            img_set.update({'label_image': label_name})

            count = 0

            for method_name, prediction_root in to_predictions.items():

                count = count + 1

                prediction_name = os.path.join(prediction_root, os.path.splitext(img_name)[0] + '.png')

                prediction = Image.open(prediction_name)

                img_set.update({str(method_name): str(prediction_name)})

                # draw = ImageDraw.Draw(prediction)

                prediction_cv = cv.cvtColor(np.asarray(prediction), cv.COLOR_RGB2BGR)

                ber_single = compute_BER(prediction, label_img)  # Use BER to eval

                iou_single = compute_IOU(prediction, label_img)  # Use IOU to eval

                # deep_dist = deep_rank.get_distance(label_name, prediction_name)  # Use Deep Rank to eval

                ber_mul_best.append(ber_single)

                iou_mul.update({str(method_name): iou_single})
                ber_mul.update({str(method_name): ber_single})

                if str(method_name) in predictions:
                    predictions[str(method_name)] = predictions[str(method_name)]+ber_single
                else:
                    predictions.update({str(method_name): ber_single})
                print(method_name, ' BER Single : ', ber_single, ' IOU : ', iou_single)#, ' Deep dist : ', deep_dist)

                # cv.imshow(method_name + ": " + str(np.around(ber_single, 3)), prediction_cv)

            sorted_iou_mul = sorted(iou_mul.items(), key=operator.itemgetter(1), reverse=True)
            sorted_ber_mul = sorted(ber_mul.items(), key=operator.itemgetter(1))

            for i in range(len(to_predictions)):
                name1 = sorted_iou_mul[i][0]
                name2 = sorted_ber_mul[i][0]
                if name1 != name2:
                    show_image(img_set)
            # while cv.waitKeyEx(0) != 32:
            #     a = 1
            #
            # cv.destroyAllWindows()

            ber_best = min(ber_mul_best)

            BER.append(ber_best)

    for method_name, prediction_root in to_predictions.items():
        predictions[str(method_name)] = predictions[str(method_name)] / img_number
        print(str(method_name), ' mean ber:  ', predictions[str(method_name)])

    BER_mean = np.nanmean(BER)

    print('BER_mean : ', BER_mean)
