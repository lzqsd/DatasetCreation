import glob
import os.path as osp
import os
import struct
import numpy as np
import cv2
import models
import torch
from torch.autograd import Variable

imWidth = 160
imHeight = 120
envHeight = 16
envWidth = 32

renderLayer = models.renderingLayer()

scenes = glob.glob(osp.join('Images', 'scene0001_*') )
scenes = sorted(scenes )
for scene in scenes:
    hdrs = glob.glob(osp.join(scene, 'imenv_*.hdr') )
    hdrs = sorted(hdrs )
    for hdr in hdrs:
        print(hdr )
        envs = cv2.imread(hdr, -1)
        envs = envs.reshape([imHeight, envHeight, imWidth, envWidth, 3 ] )
        envs = envs.transpose([0, 2, 1, 3, 4] )

        '''
        envIm = np.transpose(envs, (0, 2, 1, 3, 4)  )
        envIm = envIm[1:imHeight:20, :, 1:imWidth:20, :, :]
        envIm = envIm.reshape(6 * envHeight, 8 * envWidth, 3 )
        envIm = envIm / np.mean(envIm ) * 0.5
        envIm = np.clip(envIm, 0, 1) ** (1.0/2.2)
        envIm = (255 * envIm ).astype(np.uint8 )
        envName = hdr.replace('hdr', 'png')
        cv2.imwrite(envName, envIm[:, :, ::-1] )
        '''

        imName = hdr.replace('hdr', 'rgbe').replace('imenv_', 'im_')
        im = cv2.imread(imName, -1 )[:, :, ::-1]
        imScale = np.mean(im )

        albedoName = hdr.replace('hdr', 'png').replace('imenv_', 'imbaseColor_')
        albedo = np.ascontiguousarray(cv2.imread(albedoName )[:, :, ::-1] )

        normalName = hdr.replace('hdr', 'png').replace('imenv_', 'imnormal_')
        normal = np.ascontiguousarray(cv2.imread(normalName )[:, :, ::-1] )

        roughName = hdr.replace('hdr', 'png').replace('imenv_', 'imroughness_')
        rough = np.ascontiguousarray(cv2.imread(roughName )[:, :, 0:1] )

        albedoBatch = albedo.astype(np.float32 ).transpose([2, 0, 1])
        albedoBatch = (albedoBatch / 255.0 )[np.newaxis, :]
        albedoBatch = Variable(torch.from_numpy(albedoBatch ) ).cuda(0)

        normalBatch = normal.astype(np.float32 ).transpose([2, 0, 1])
        normalBatch = (normalBatch / 127.5 - 1 )[np.newaxis, :]
        normalBatch = normalBatch / np.maximum(np.sqrt(np.sum(normalBatch * normalBatch,
            axis=1)[:,  np.newaxis, :] ), 1e-6 )
        normalBatch = Variable(torch.from_numpy(normalBatch ) ).cuda(0)

        roughBatch = rough.astype(np.float32 ).transpose([2, 0, 1])
        roughBatch = (roughBatch / 127.5 - 1 )[np.newaxis, :]
        roughBatch = Variable(torch.from_numpy(roughBatch ) ).cuda(0)

        envBatch = envs.transpose([4, 0, 1, 2, 3] )
        envBatch = envBatch[np.newaxis, :]
        envBatch = Variable(torch.from_numpy(envBatch ) ).cuda(0)

        diffuseIm, specularIm = renderLayer.forwardEnv(albedoBatch, normalBatch, roughBatch, envBatch )
        renderIm = (diffuseIm + specularIm).data.cpu().squeeze().numpy()
        renderIm = renderIm.transpose([1, 2, 0])
        renderIm = renderIm / imScale * 0.5
        renderIm = np.clip(renderIm, 0, 1 )
        renderIm = (255 * (renderIm ** (1.0/2.2) ) ).astype(np.uint8 )
        renderImName = hdr.replace('hdr', 'png').replace('imenv_', 'imrendered_')
        cv2.imwrite(renderImName, renderIm[:, :, ::-1] )


