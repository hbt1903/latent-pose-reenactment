pip3 install \
    torch==1.5.1 \
    torchvision==0.6.1

pip3 install \
    tensorboard==2.2.2 \
    pandas==1.0.5 \
    torchsummary==1.5.1 \
    thop==0.0.31-2005241907 \
    tqdm==4.47.0 \
    scipy==1.5.1 \
    opencv-python==4.3.0.36 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    pymemcache==3.2.0 \
    python-memcached==1.59 \
    matplotlib==3.2.2 \
    seaborn==0.10.1 \
    huepy==1.2.1 \
    albumentations==0.4.6 \
    imgaug==0.4.0 \
    PyYAML==5.3.1 \
    munch==2.5.0 \
    pickle5==0.0.11 \
    jpeg4py==0.1.4 \
    face-alignment==1.1.1 \
    mxnet-cu101mkl==1.6.0.post0 \
    scikit-learn==0.23.2

pip3 install git+https://github.com/DmitryUlyanov/yamlenv.git
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip3 install git+https://github.com/dniku/insightface.git@763060b2

git clone https://github.com/shrubb/latent-pose-reenactment
git clone https://github.com/shrubb/Graphonomy.git latent-pose-reenactment/utils/Graphonomy

pip install --upgrade --no-cache-dir gdown

gdown --no-cookies --id 1QIVDRgtc_Fkaz9M4kewLU-TAFwJm7fvT
gdown --no-cookies --id 1YDZRDnrPdJYvyRhe_Ir-91OFR_TCwvgO
gdown --no-cookies --id 1o7s95QCrhRGyomuJDp5uGF1rhFSrbBN9
gdown --no-cookies --id 1_ia1zRNZbfozBLVy-iOxYxdWRhtGhmZ2

mkdir latent-pose-reenactment/model
mkdir latent-pose-reenactment/utils/Graphonomy/data/model

mv latent-pose-release.pth latent-pose-reenactment/model/latent-pose-release.pth
mv universal_trained.pth latent-pose-reenactment/utils/Graphonomy/data/model/universal_trained.pth
mv vgg19-d01eb7cb.pth latent-pose-reenactment/criterions/common/vgg19-d01eb7cb.pth
mv vgg_face_weights.pth latent-pose-reenactment/criterions/common/vgg_face_weights.pth
mv images latent-pose-reenactment/
mv videos latent-pose-reenactment/
mv inference_folder.py latent-pose-reenactment/utils/Graphonomy/exp/inference/
rm latent-pose-reenactment/utils/preprocess_dataset.sh
mv preprocess_dataset.sh latent-pose-reenactment/utils/
mv meta_train.sh latent-pose-reenactment/utils/

chmod +x latent-pose-reenactment/utils/meta_train.sh
chmod +x latent-pose-reenactment/utils/preprocess_dataset.sh
