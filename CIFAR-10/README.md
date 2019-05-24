# CIFAR-10 models

##First model: simple CNN used as a baseline. Taken from :

##Second model: More complex CNN used to get improved result.
Architecture:
|          Layer (type)          	|    Output Shape    	| Param # 	|
|:------------------------------:	|:------------------:	|:-------:	|
| conv2d_1 (Conv2D)              	| (None, 32, 32, 32) 	| 896     	|
| activation_1 (Activation)      	| (None, 32, 32, 32) 	| 0       	|
| batch_normalization_1 (Batch)  	| (None, 32, 32, 32) 	| 128     	|
|:------------------------------:	|:------------------:	|:-------:	|
| conv2d_2 (Conv2D)              	| (None, 32, 32, 32) 	| 9248    	|
| activation_2 (Activation)      	| (None, 32, 32, 32) 	| 0       	|
| batch_normalization_2 (Batch)  	| (None, 32, 32, 32) 	| 128     	|
| max_pooling2d_1 (MaxPooling2)  	| (None, 16, 16, 32) 	| 0       	|
| dropout_1 (Dropout)            	| (None, 16, 16, 32) 	| 0       	|
|:------------------------------:	|:------------------:	|:-------:	|
| conv2d_3 (Conv2D)              	| (None, 16, 16, 64) 	| 18496   	|
| activation_3 (Activation)      	| (None, 16, 16, 64) 	| 0       	|
| batch_normalization_3 (Batch)  	| (None, 16, 16, 64) 	| 256     	|
|:------------------------------:	|:------------------:	|:-------:	|
| conv2d_4 (Conv2D)              	| (None, 16, 16, 64) 	| 36928   	|
| activation_4 (Activation)      	| (None, 16, 16, 64) 	| 0       	|
| batch_normalization_4 (Batch)  	| (None, 16, 16, 64) 	| 256     	|
| max_pooling2d_2 (MaxPooling2)  	| (None, 8, 8, 64)   	| 0       	|
| dropout_2 (Dropout)            	| (None, 8, 8, 64)   	| 0       	|
|:------------------------------:	|:------------------:	|:-------:	|
| conv2d_5 (Conv2D)              	| (None, 8, 8, 128)  	| 73856   	|
| activation_5 (Activation)      	| (None, 8, 8, 128)  	| 0       	|
| batch_normalization_5 (Batch)  	| (None, 8, 8, 128)  	| 512     	|
|:------------------------------:	|:------------------:	|:-------:	|
| conv2d_6 (Conv2D)              	| (None, 8, 8, 128)  	| 147584  	|
| activation_6 (Activation)      	| (None, 8, 8, 128)  	| 0       	|
| batch_normalization_6 (Batch)  	| (None, 8, 8, 128)  	| 512     	|
| max_pooling2d_3 (MaxPooling2)  	| (None, 4, 4, 128)  	| 0       	|
| dropout_3 (Dropout)            	| (None, 4, 4, 128)  	| 0       	|
|:------------------------------:	|:------------------:	|:-------:	|
| flatten_1 (Flatten)            	| (None, 2048)       	| 0       	|
|:------------------------------:	|:------------------:	|:-------:	|
| dense_1 (Dense)                	| (None, 10)         	| 20490   	|

  Total params: 309,290
  Trainable params: 308,394
  Non-trainable params: 896
