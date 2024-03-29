## [3.0.0](https://github.com/ndido98/revelio/compare/2.0.0...3.0.0) (2023-01-09)


### ⚠ BREAKING CHANGES

* **neuralnet:** make get_state_dict return deep copy

### Features

* **augmentation:** add jpeg2000 compression step ([6700c95](https://github.com/ndido98/revelio/commit/6700c95b18b0ec53faaead64062926cbee6e6466))
* **caching:** abstract face detection/feature extraction load/save with cachers ([8994668](https://github.com/ndido98/revelio/commit/8994668756ecf17b624e18115ed5f4e65dd54296))
* **feature-extraction:** add stationary wavelet packets extraction ([905c78a](https://github.com/ndido98/revelio/commit/905c78a347c33b8b690ffee831cf0541fef556db))
* **nn:** add load_from_checkpoint argument to optimizer when fine tuning ([32c41d5](https://github.com/ndido98/revelio/commit/32c41d58620c4d70aecb1f0ae1cac2d68122b1e3))
* **utils:** add glob_multiple utility function ([51328a3](https://github.com/ndido98/revelio/commit/51328a3ed6bf9f0e40aab854bea90fdb83891829))


### Bug Fixes

* **cli:** add pretty printing of exceptions when not verbose ([26a374d](https://github.com/ndido98/revelio/commit/26a374da5d4d8e213c44c2a583e3684d023f9e39))
* **dataset:** allow data augmentation to skip failing step ([f5e6b96](https://github.com/ndido98/revelio/commit/f5e6b965ccfa94e78f9fbc04c526b9a138611b35))
* **neuralnet:** make get_state_dict return deep copy ([7236c50](https://github.com/ndido98/revelio/commit/7236c5070470e27c876ba6019b84423a11136336))

## [2.0.0](https://github.com/ndido98/revelio/compare/1.0.6...2.0.0) (2022-12-12)


### ⚠ BREAKING CHANGES

* **augmentation:** change default N2 noise value for print&scan simulation
* **dataset:** remove automatic bgr to rgb conversion
* **cli:** reset seed before training and before evaluation
* **augmentation:** change signature of augmentation process_element

### Features

* **augmentation:** add grayscale augmentation ([b81d2db](https://github.com/ndido98/revelio/commit/b81d2dbd02aed47963bf131f31a0ee99c54c984b))
* **augmentation:** add jpeg and resize augmentations ([5b52035](https://github.com/ndido98/revelio/commit/5b520355b6672727abd709882e4dee566a34f2cd))
* **augmentation:** add print&scan augmentation step ([458394b](https://github.com/ndido98/revelio/commit/458394b15a193f4264debd8fb847202a896b9158))
* **augmentation:** add stack config to grayscale step ([a79e642](https://github.com/ndido98/revelio/commit/a79e642dcef14765eaee1c498d3917cf3f668a57))
* **config:** add json metrics file report ([4032d95](https://github.com/ndido98/revelio/commit/4032d95d462d464cc5a173ee1505829d4fc868b5))
* **dataset:** allow for empty test set ([ef11f07](https://github.com/ndido98/revelio/commit/ef11f071e55bc5cce249441e0b2f205a97601828))
* **feature-extraction:** add prnu, fourier and wavelets ([4393e3e](https://github.com/ndido98/revelio/commit/4393e3e11934c46a775f38e55d50bc94aaf764c1))
* **loaders:** add cfd/cfdmorph loaders ([4454797](https://github.com/ndido98/revelio/commit/4454797cd57ed4d99bc41a6cd2b0ffe882e6464c))
* **loaders:** add morph level args to CFDMorph loader ([f273cc5](https://github.com/ndido98/revelio/commit/f273cc5f3e1513440472883921cf94df1e438167))
* **loaders:** add png and jpg loading for cfd and cfdmorph ([0106e99](https://github.com/ndido98/revelio/commit/0106e990510f53138ed7447b1b154a9f3e17697d))
* **model:** add feature inception resnet ([d247360](https://github.com/ndido98/revelio/commit/d247360ddeb902c12d6f77d5832164774fbd777f))
* **model:** use 8 instead of 5 decimals in scores ([5b818d4](https://github.com/ndido98/revelio/commit/5b818d4d4bd1f3bfd48df53988e2a14c82e689e3))
* **preprocessing:** add dataset-variant preprocessing ([daa6247](https://github.com/ndido98/revelio/commit/daa6247857997a37f966a7c1f694c7841c6bf181))
* **preprocessing:** add select channel and color space conversions ([b8795b9](https://github.com/ndido98/revelio/commit/b8795b9136d419fb8ef9d20099c7214def8a0495))
* **preprocessing:** make maximum value configurable for each channel ([ef58d51](https://github.com/ndido98/revelio/commit/ef58d51238beeb518fe8232bcd99385123847c82))


### Bug Fixes

* **augmentation:** change default N2 noise value for print&scan simulation ([428957d](https://github.com/ndido98/revelio/commit/428957dd0f75e07e357292eef3f156e6611e187e))
* **augmentation:** make print&scan sigma proportional to image diagonal ([4ce23e1](https://github.com/ndido98/revelio/commit/4ce23e193d57778ecc2c0ae4ae791dc7b1046e48))
* **callbacks:** fix missing image report if fine tuning ([9b7abb6](https://github.com/ndido98/revelio/commit/9b7abb615ce9c8600b6120fd64d01a4e2aba7015))
* **cli:** reset seed before training and before evaluation ([3d8d62b](https://github.com/ndido98/revelio/commit/3d8d62b49270ea3b11537cd1c15c4af99e5b228a))
* **dataset:** fix use before assign in offline processing ([677d798](https://github.com/ndido98/revelio/commit/677d798999d766326ea2e5d20565d559286ef9c8))
* **model:** mitigate memory leak in model evaluation ([a7919b6](https://github.com/ndido98/revelio/commit/a7919b6bace16c4e7317bd29c27b02779d1c7c2d))


### Code Refactoring

* **augmentation:** change signature of augmentation process_element ([cab004f](https://github.com/ndido98/revelio/commit/cab004fbe945b3685e3fb850a0683e613fe63cbc))
* **dataset:** remove automatic bgr to rgb conversion ([5e28a29](https://github.com/ndido98/revelio/commit/5e28a29bff0355517c1e7a323a8d2a39de741b79))

## [1.0.6](https://github.com/ndido98/revelio/compare/1.0.5...1.0.6) (2022-12-08)


### Bug Fixes

* **deps:** update dependency scikit-learn to ~1.2.0 ([c856597](https://github.com/ndido98/revelio/commit/c856597e1f9a8ddb69dce3933bd0219747668620))

## [1.0.5](https://github.com/ndido98/revelio/compare/1.0.4...1.0.5) (2022-12-07)


### Bug Fixes

* **deps:** update dependency wandb to v0.13.6 ([65436b3](https://github.com/ndido98/revelio/commit/65436b3e74af171605f9a38a1e6c81e2f6a91ce9))

## [1.0.4](https://github.com/ndido98/revelio/compare/1.0.3...1.0.4) (2022-11-20)


### Bug Fixes

* **deps:** update dependency numpy to v1.23.5 ([061aea3](https://github.com/ndido98/revelio/commit/061aea3c8ea07c20ad24a6ebb8246b21e71d0b65))

## [1.0.3](https://github.com/ndido98/revelio/compare/1.0.2...1.0.3) (2022-11-09)


### Bug Fixes

* **deps:** update dependency tensorboard to ~2.11.0 ([c2bdf22](https://github.com/ndido98/revelio/commit/c2bdf2283e2daf269ca1a97745766bb47b4dbb79))

## [1.0.2](https://github.com/ndido98/revelio/compare/1.0.1...1.0.2) (2022-11-04)


### Bug Fixes

* **deps:** update dependency wandb to v0.13.5 ([2fa7a78](https://github.com/ndido98/revelio/commit/2fa7a7836f676e43875b43be287fdbce7098bd24))

## [1.0.1](https://github.com/ndido98/revelio/compare/1.0.0...1.0.1) (2022-11-03)


### Bug Fixes

* **deps:** update dependency matplotlib to v3.6.2 ([e192a41](https://github.com/ndido98/revelio/commit/e192a4163e70a1b3b903972c747c8c99d3801cc0))

## 1.0.0 (2022-11-02)


### Features

* add logging ([474f39f](https://github.com/ndido98/revelio/commit/474f39f0419ef88282a4a7e3d89a387a56dd1d56))
* **augmentation:** add applies_to field ([8a1a21e](https://github.com/ndido98/revelio/commit/8a1a21e03ffd56b6f73fdd5d4cbf4d032a827e06))
* **augmentation:** add augmentation steps ([2eb2b8e](https://github.com/ndido98/revelio/commit/2eb2b8ef0e48c6ac8f7c75f05f13ed1f4822713f))
* **callbacks:** add early stopping ([ad48f03](https://github.com/ndido98/revelio/commit/ad48f033de95f635d82dd12a5f9e1a9e2e94af2f))
* **callbacks:** add memory profiling to tensorboard ([5fdd28f](https://github.com/ndido98/revelio/commit/5fdd28fc8a7832f0114472c8d217d7cbee50a384))
* **callbacks:** add model checkpoint ([b5ffef8](https://github.com/ndido98/revelio/commit/b5ffef815de8d4043d86e02915edfdfc7b04afeb))
* **callbacks:** add steps count ([13b7ea4](https://github.com/ndido98/revelio/commit/13b7ea49b7e972dc2971f8cacac6f2417e105b07))
* **callbacks:** add tensorboard batch viz, graph ([c7c868e](https://github.com/ndido98/revelio/commit/c7c868e06c396fdcdcca074786e34484b700bc3f))
* **callbacks:** add tensorboard callback ([a86ddeb](https://github.com/ndido98/revelio/commit/a86ddeb9b254455a149dedca8ba3ec5f5bff5c6d))
* **cli:** add --no-warmup argument to skip warmup ([c9fce89](https://github.com/ndido98/revelio/commit/c9fce895a816cdde4d23be98439ef67c84b90716))
* **cli:** add cli argparser ([ac5cc8c](https://github.com/ndido98/revelio/commit/ac5cc8c242b1539d9e28ce3e0033ecb7c3047fa1))
* **cli:** add configurable warmup workers count ([171bc21](https://github.com/ndido98/revelio/commit/171bc212ce04b751a97f04774326c52f37dfc1f0))
* **cli:** add model fitting to cli ([21e0d0c](https://github.com/ndido98/revelio/commit/21e0d0c61cc6888a9885355424341af7a299daa3))
* **cli:** avoid creating train/val workers when only inferencing ([0258090](https://github.com/ndido98/revelio/commit/0258090a27b34ecf1f80f90d79340e792d2ba4d5))
* **config:** add config model ([6060d89](https://github.com/ndido98/revelio/commit/6060d8900a36113a413f95ee3f05265b06f6b7c5))
* **config:** add configurable seed ([0f43c1a](https://github.com/ndido98/revelio/commit/0f43c1a2be2d7c97f425ce885e80dffadd0ed38a))
* **config:** add templating to scores files path ([1a2c3e5](https://github.com/ndido98/revelio/commit/1a2c3e56352ad056c57888212be1e28e3caded76))
* **dataset:** add check to make sure all elements have the same number of x ([97993c9](https://github.com/ndido98/revelio/commit/97993c9c046dfe7ed3efe410d892a1112ebb40fa))
* **dataset:** add dataset element object ([4754cfb](https://github.com/ndido98/revelio/commit/4754cfbff0323519708617eecce8bba5adc9b5be))
* **dataset:** add dataset factory and torch dataset ([c1080f9](https://github.com/ndido98/revelio/commit/c1080f918ad7d8fb8372cd498e63e04924a7dca8))
* **dataset:** add dataset loader ([9627071](https://github.com/ndido98/revelio/commit/962707158978e086f1d2325bb5d524f390c7e800))
* **dataset:** add explicit loader with args ([e105f53](https://github.com/ndido98/revelio/commit/e105f534561dd32b6cfd95521a38598bcc670013))
* **dataset:** add length ([fec1bae](https://github.com/ndido98/revelio/commit/fec1bae861a5f447f329e7603e1cdc4c4223fd9f))
* **dataset:** add stats printing ([8bc2a73](https://github.com/ndido98/revelio/commit/8bc2a73598bd48ccc1a8e400a06c1092b5d400d8))
* **dataset:** add warmup function ([3b78208](https://github.com/ndido98/revelio/commit/3b78208607b743e94182d1f9bab00588957b6c9e))
* **dataset:** create testing groups and rework splitting ([ac433b3](https://github.com/ndido98/revelio/commit/ac433b3479bc2295d4f60827490b34be6c1b4bfc))
* **face-detection:** add dlib detector ([b78add1](https://github.com/ndido98/revelio/commit/b78add17480c7f0fe88b77e1265b65e5d96d7649))
* **face-detection:** add face detector module ([3767762](https://github.com/ndido98/revelio/commit/37677620e628ae63ff2212fa301800f3a08c4965))
* **face-detection:** add opencv and mtcnn detectors ([cf0cc4e](https://github.com/ndido98/revelio/commit/cf0cc4ed773ab01ff6524c9d5d1e46cb92556eea))
* **feature-extraction:** add feature extractors ([2dafd04](https://github.com/ndido98/revelio/commit/2dafd04f09386b631476a173dbddae69a74e49d3))
* **loaders:** add biometix morphed loader ([1ecc55a](https://github.com/ndido98/revelio/commit/1ecc55a2a7315bbe9231bbf6107f5aa85b172117))
* **loaders:** add morphdb loader ([6de92b8](https://github.com/ndido98/revelio/commit/6de92b8cd85776d68e489d5c788e5cd3eae8f4d1))
* **loaders:** add pmdb loader ([a74c60b](https://github.com/ndido98/revelio/commit/a74c60b5cfc6c53eb6863d6a41c9d57b39d89c4e))
* **loaders:** add several loaders ([553f258](https://github.com/ndido98/revelio/commit/553f258ef16a7bcec51a33261642670a38f5fd7b))
* **losses:** add adam ([56992b3](https://github.com/ndido98/revelio/commit/56992b38950b15c62e4612c10964376e9d0b8aeb))
* **metrics:** add accuracy ([2ebd862](https://github.com/ndido98/revelio/commit/2ebd8627d59d2da443a1ba8a062446265b4cda29))
* **metrics:** add eer and bpcer@apcer ([0f25bc7](https://github.com/ndido98/revelio/commit/0f25bc7d0cdb2012d0174bfb46a08e0867832932))
* **metrics:** add epoch_* metrics for tensorboard, checkpoint and early stopping callbacks ([212351a](https://github.com/ndido98/revelio/commit/212351a4d77dda8019a18df146e7e76447e5752d))
* **metrics:** add metrics ([7938b00](https://github.com/ndido98/revelio/commit/7938b00cd05893917ccd50cf2d7256a24ef0cd0a))
* **metrics:** add tpr and tnr ([cea5757](https://github.com/ndido98/revelio/commit/cea5757b50125c52a0dc8b2233d488db9125a859))
* **metrics:** allow multiple values in one metric ([e45cddd](https://github.com/ndido98/revelio/commit/e45cdddfe90d8df04b07e7802f94d4d43be0cb21))
* **metrics:** expose device in which metrics are run ([c50a2fc](https://github.com/ndido98/revelio/commit/c50a2fca2121d38ce79ea61470fa0502dd208971))
* **model:** add alexnet, vgg and resnet ([81b603f](https://github.com/ndido98/revelio/commit/81b603f416a34f2f5992991b00ca82a625500385))
* **model:** add base model class ([ded81c9](https://github.com/ndido98/revelio/commit/ded81c99ee86dc3c14cdcc2e9ef91a34c58ab616))
* **model:** add inception resnet ([6fbe362](https://github.com/ndido98/revelio/commit/6fbe362be4cf239493fb3d81b2e84e077ebd138b))
* **model:** add mobilenet ([cbdfa5f](https://github.com/ndido98/revelio/commit/cbdfa5f08399a00ccd0138bbf711739d89623eba))
* **model:** add neural network class ([4d0eff2](https://github.com/ndido98/revelio/commit/4d0eff24ee32d00b596c9eb6c946050eb9f7cccb))
* **model:** add random guesser ([a4dd65a](https://github.com/ndido98/revelio/commit/a4dd65af3320236aa55e75075b9584c2d50b581d))
* **model:** add save/load checkpoint ([097c78f](https://github.com/ndido98/revelio/commit/097c78f8c95a580926981394bbe337beb8a3ea0f))
* **model:** add squeezenet ([7e6b44d](https://github.com/ndido98/revelio/commit/7e6b44d36b213ca707485081114a26d0a874c825))
* **model:** add vision transformer ([1c441ff](https://github.com/ndido98/revelio/commit/1c441ff96bf0d1a46d272db69448ef03c0ac1eda))
* **model:** move scores file path eval to model score computation ([ecf21f8](https://github.com/ndido98/revelio/commit/ecf21f8f78db1f0486ea25ab41d19bb6f00973e9))
* **optimizers:** add binary cross entropy ([3f34637](https://github.com/ndido98/revelio/commit/3f34637f69590899a59c278d622921f6c72be8d8))
* **preprocessing:** add normalization preprocessing ([230650a](https://github.com/ndido98/revelio/commit/230650a04a05578a936f2c43aac0635272dce6b9))
* **preprocessing:** add preprocessing phase after feature extraction ([47c35e4](https://github.com/ndido98/revelio/commit/47c35e448d109a23579684672d597188fe6ea249))
* **preprocessing:** add uint8 -> float32 preprocessing ([24bf94b](https://github.com/ndido98/revelio/commit/24bf94b3cea4ba6faba5a606d1dd50cc87fe8510))
* **registry:** add - as ignored char ([1b395c7](https://github.com/ndido98/revelio/commit/1b395c71bc9aa354717f4be20a87414cf227b0dd))
* **registry:** add transparent registrable classes ([3da9dd6](https://github.com/ndido98/revelio/commit/3da9dd690c4010619611c1f11f024b9cee423912))
* **registry:** allow snake case names ([9925ad9](https://github.com/ndido98/revelio/commit/9925ad92b98e13c3ba24d210871547e885f23109))
* **registry:** make kwargs with _ assignable only explicitly ([906c73c](https://github.com/ndido98/revelio/commit/906c73ccecf3814d059ec9a28c0ebb4ce0d95cdf))


### Bug Fixes

* add dataset root to dataset element ([c07e304](https://github.com/ndido98/revelio/commit/c07e3045701e72a80be2b45cb69f98161d44a0ec))
* **augmentation:** change signature of step to take only the image ([bb161e2](https://github.com/ndido98/revelio/commit/bb161e25cd66c0984a4f41586fd04f9ac382c827))
* **callbacks:** add mkdir to model checkpoint target directory ([dc8c170](https://github.com/ndido98/revelio/commit/dc8c170ceb0c00048fa729c670f674ab02a63a0f))
* **callbacks:** change bona fide to live image reporting ([f25c69b](https://github.com/ndido98/revelio/commit/f25c69b7d05ce7caab8f79c4cb44ef6eefc65d71))
* **callbacks:** fix tensorboard graph/image display ([598ef5a](https://github.com/ndido98/revelio/commit/598ef5a9cb0d8f8141df76417dede7b26beb19bb))
* **callbacks:** import early stopping ([d097f89](https://github.com/ndido98/revelio/commit/d097f89191c01be542b57b24c3c01941ed4daea2))
* **callbacks:** remove tensorboard graph ([b3ed308](https://github.com/ndido98/revelio/commit/b3ed308429fe4646f99e39c84eecac13128e37f8))
* **cli:** create dataloaders just for warming up for better progress reporting ([bf77bb7](https://github.com/ndido98/revelio/commit/bf77bb74a1171dd0b13bb5f9154e7957bb10385c))
* **cli:** disable persistent workers if no workers are used ([8ff3170](https://github.com/ndido98/revelio/commit/8ff3170e3db4856414499b638d5bbbe2fb903969))
* **cli:** use consume to warm up the datasets ([4a302ef](https://github.com/ndido98/revelio/commit/4a302eff0256f8e52cb9f5476690bec968cd9ea1))
* **config:** allow for no preprocessing ([5df3bfc](https://github.com/ndido98/revelio/commit/5df3bfc382fd3369b110af38fbdd1d6296a7da43))
* **config:** change DirectoryPath to str for yet-to-be directories ([b096dbf](https://github.com/ndido98/revelio/commit/b096dbf96256e83442eddddbc29274ba78b37958))
* **config:** fix arg name cannot start with underscore ([9745aa3](https://github.com/ndido98/revelio/commit/9745aa3aaa77f4abd9155fc49143bb9d14813ee9))
* **config:** make args default to empty ([3ddb895](https://github.com/ndido98/revelio/commit/3ddb895c06b4a3e2af8d13b68e0449424fceb871))
* **config:** validate paths without checking their existence ([397bad0](https://github.com/ndido98/revelio/commit/397bad0b0a5fb8761a8ae91ab6fe33d407aa89f3))
* **dataset:** add missing y label to yielded element ([cb98631](https://github.com/ndido98/revelio/commit/cb986312a6d561a1e0921245ecb2720ebf1c4ff0))
* **dataset:** add randomization of dataset at each epoch ([7842a32](https://github.com/ndido98/revelio/commit/7842a32c56ea6b11e5d90ef158c799413093f334))
* **dataset:** allow float32 images ([755c1a2](https://github.com/ndido98/revelio/commit/755c1a2087d6571703ad8a27d07ae29ed644512a))
* **dataset:** apply color and channel transposion ([3e9f853](https://github.com/ndido98/revelio/commit/3e9f8532b15446d19bd8334d8f663ee6507405b4))
* **dataset:** force gc collection if not loaded from cache ([45e1353](https://github.com/ndido98/revelio/commit/45e1353ebf3cd0951523b6d99af401131ef39b69))
* **dataset:** make face detection offline ([8c97c30](https://github.com/ndido98/revelio/commit/8c97c306254b627d2a70f938262b6322427589cf))
* **dataset:** remove offline processing when not warming up ([eb6ae71](https://github.com/ndido98/revelio/commit/eb6ae711c07575bb5332c88eadfb2932c98d9dd8))
* **dataset:** remove warmup function and instead use boolean flag ([28b42de](https://github.com/ndido98/revelio/commit/28b42de03367be8dda3d03b5ed2a5e1e95f33d8c))
* **dataset:** skip elements if face detection or feature extraction fails ([d67702c](https://github.com/ndido98/revelio/commit/d67702cfa8fae5431f54b5c3e99aa79fadfa0833))
* **dataset:** use specialized list to avoid memory leaks ([cd3b3ac](https://github.com/ndido98/revelio/commit/cd3b3ac93520a70ff1f506c4ee978f53e6938e87))
* **deps:** update dependency matplotlib to v3.6.1 ([429ca2e](https://github.com/ndido98/revelio/commit/429ca2efa27a80606d6f9aaf6ea766a28ea5a9d5))
* **deps:** update dependency numpy to v1.23.4 ([70ab22f](https://github.com/ndido98/revelio/commit/70ab22f5bb9528f56a3182e6a68ed6a31a2acd14))
* **deps:** update dependency scikit-learn to v1.1.3 ([cfc8a7e](https://github.com/ndido98/revelio/commit/cfc8a7e364f4cb74a6725d916a6f10b8a1fc64ed))
* **deps:** update dependency scipy to v1.9.2 ([7c84265](https://github.com/ndido98/revelio/commit/7c842659584eb4d9f9cd2f1ddd078b341ab90d2c))
* **deps:** update dependency scipy to v1.9.3 ([5b49637](https://github.com/ndido98/revelio/commit/5b49637aa85863b769378c4deb75c34221be37a8))
* **face-detection:** clip bounding box inside image ([cfadaca](https://github.com/ndido98/revelio/commit/cfadaca409610a3ce3f5c66183d1308b628a2b51))
* **face-detection:** fix numpy arrays not json serializable ([714ef60](https://github.com/ndido98/revelio/commit/714ef608d363cc059d6c0b484ff284b4f1498940))
* **face-detection:** take biggest bounding box for opencv/mtcnn multiple results ([8d96535](https://github.com/ndido98/revelio/commit/8d9653539377646dfdb879da06366dd861a6ab4e))
* **loaders:** make amsl loader deterministic ([5e8b92f](https://github.com/ndido98/revelio/commit/5e8b92f8c683022d6a3b4c43c17966f2b0dffc8d))
* **metrics:** adapt bpcer@apcer to be more lax ([2374786](https://github.com/ndido98/revelio/commit/23747862daf39c223c61c80833b9a76fd7eaf6fc))
* **metrics:** add conditional to remove nan cases ([5158c87](https://github.com/ndido98/revelio/commit/5158c877d7aee05cfb1ace30ae67844dcbce9347))
* **metrics:** improve display of accuracy and bpcer@apcer ([4a68287](https://github.com/ndido98/revelio/commit/4a68287dc543bf795a01854388b834acaf1795ef))
* **metrics:** use abstract property for name ([acad600](https://github.com/ndido98/revelio/commit/acad600e29305faf7f301dd6eb617a778b64e882))
* **model:** add list case to _dict_to_device and fix prediction scores accumulation ([a8f6c47](https://github.com/ndido98/revelio/commit/a8f6c473ea31bc204574887b67e6e444c817554e))
* **model:** add missing definition of resnet model if pretrained ([ef090a7](https://github.com/ndido98/revelio/commit/ef090a7266c9e486a10f284feafc8256b8b63d0f))
* **model:** apply sigmoid to logits output, remove cumulative loss ([5b3b2d6](https://github.com/ndido98/revelio/commit/5b3b2d648a4b34eb0971f93fce91cca0c8e2fb4d))
* **model:** change scores file format ([6bc5b9d](https://github.com/ndido98/revelio/commit/6bc5b9dd9612fd59ffe47469735b6ab963029cb8))
* **model:** fix epoch loading from state dict ([c32ee11](https://github.com/ndido98/revelio/commit/c32ee110d2e15df9d23f4a6432a862d9878b096d))
* **model:** import neural nets module for registration ([08fc220](https://github.com/ndido98/revelio/commit/08fc220ae58476c178ba53a49790b1f51dd0f46b))
* **model:** load metrics from model constructor ([445cdff](https://github.com/ndido98/revelio/commit/445cdff79350d215b054adb97a8ac0ce728dc93e))
* **model:** move metrics reset outside batch processing ([bbfbb68](https://github.com/ndido98/revelio/commit/bbfbb68c840bbf7c8689a5f9d2abca07764b3c61))
* **model:** move predictions to correct device when computing metrics ([8f5df94](https://github.com/ndido98/revelio/commit/8f5df948558fb83a7584e1ec3165156c8bc22801))
* **nn:** don't load callbacks if not training ([c6879b2](https://github.com/ndido98/revelio/commit/c6879b2c73b95d16c882f154eca972780ab1b872))
* **preprocessing:** add interpolation to resize ([7ee7580](https://github.com/ndido98/revelio/commit/7ee7580dd5d31bdaf3b313d69ead99baedd14654))
* **preprocessing:** redo args validation for normalize ([db7e546](https://github.com/ndido98/revelio/commit/db7e546026279d72cdc5e5d9cbb71a3fd236fa5c))
* **registry:** fix bug when loading class with args ([365d942](https://github.com/ndido98/revelio/commit/365d94252ef93efed62147f68c9a7432b10fd9a8))
* **registry:** move args sanitization to config ([a690b80](https://github.com/ndido98/revelio/commit/a690b80b9c081aeef288ac8071bd6cada0b898a5))
* use more accurate way of counting steps in data loader ([c726320](https://github.com/ndido98/revelio/commit/c726320018c6e2d36e7bdc53d0feba1d8ee86a00))
