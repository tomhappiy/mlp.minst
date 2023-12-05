{
    "cells": [
    {
        "cell_type": "markdown",
            "source" : [
                "## 导入库"
            ] ,
            "metadata" : {
            "collapsed": false
        },
            "id" : "c15db774b543667"
    },
  {
   "cell_type": "code",
   "execution_count" : 64,
   "id" : "16eabee26f3a0bb1",
   "metadata" : {
    "collapsed": true,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:42:52.076213100Z",
     "start_time" : "2023-11-07T11:42:52.019852Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type" : "stream",
     "text" : [
      "4.5.4\n"
     ]
    }
   ] ,
   "source": [
    "from types import FunctionType\n",
    "from typing import Any, List\n",
    "from keras import Input, Model\n",
    "from keras.models import Sequential\n",
    "from keras.src.datasets.mnist import load_data\n",
    "from keras.src.layers import Activation, Dense, Flatten\n",
    "import tensorflow\n",
    "from matplotlib import pyplot\n",
    "from keras.models import load_model\n",
    "from numpy import dtype, ndarray"
   ]
  },
  {
   "cell_type": "markdown",
   "source" : [
    "## 检测GPU状态"
   ] ,
   "metadata" : {
    "collapsed": false
   },
   "id" : "1c97df5dff7ec573"
  },
  {
   "cell_type": "code",
   "execution_count" : 41,
   "outputs" : [
    {
     "name": "stdout",
     "output_type" : "stream",
     "text" : [
      "[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]\n"
     ]
    }
   ] ,
   "source": [
    "print(tensorflow.config.list_physical_devices('GPU'))"
   ] ,
   "metadata" : {
    "collapsed": false,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:14:27.114910400Z",
     "start_time" : "2023-11-07T11:14:27.108129400Z"
    }
   },
   "id": "4a4f473e13a2d13b"
  },
  {
   "cell_type": "markdown",
   "source" : [
    "## 创建模型结构"
   ] ,
   "metadata" : {
    "collapsed": false
   },
   "id" : "6fcb9fa6418a6db9"
  },
  {
   "cell_type": "code",
   "execution_count" : 42,
   "outputs" : [
    {
     "name": "stdout",
     "output_type" : "stream",
     "text" : [
      "Model: \"sequential_8\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " flatten_6 (Flatten)         (None, 784)               0         \n",
      "                                                                 \n",
      " dense_24 (Dense)            (None, 512)               401920    \n",
      "                                                                 \n",
      " activation_24 (Activation)  (None, 512)               0         \n",
      "                                                                 \n",
      " dense_25 (Dense)            (None, 512)               262656    \n",
      "                                                                 \n",
      " activation_25 (Activation)  (None, 512)               0         \n",
      "                                                                 \n",
      " dense_26 (Dense)            (None, 512)               262656    \n",
      "                                                                 \n",
      " activation_26 (Activation)  (None, 512)               0         \n",
      "                                                                 \n",
      " dense_27 (Dense)            (None, 10)                5130      \n",
      "                                                                 \n",
      " activation_27 (Activation)  (None, 10)                0         \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 932362 (3.56 MB)\n",
      "Trainable params: 932362 (3.56 MB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model: Sequential = Sequential()\n",
    "model.add(Input((28, 28)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(units = 512))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(Dense(units = 512))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(Dense(units = 512))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(Dense(units = 10))\n",
    "model.add(Activation(\"softmax\")  )\n",
    "model.summary()"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:14:27.175438300Z",
     "start_time" : "2023-11-07T11:14:27.112211600Z"
    }
   },
   "id": "1492f905ae32973"
  },
  {
   "cell_type": "markdown",
   "source" : [
    "## 加载数据"
   ] ,
   "metadata" : {
    "collapsed": false
   },
   "id" : "e06403b3aca72c74"
  },
  {
   "cell_type": "code",
   "execution_count" : 43,
   "outputs" : [] ,
   "source" : [
    "from numpy import ndarray\n",
    "\n",
    "train_X: ndarray\n",
    "train_Y: ndarray\n",
    "test_X: ndarray\n",
    "test_Y: ndarray\n",
    "(train_X, train_Y), (test_X, test_Y) = load_data()\n",
    "\n",
    "print(\"train_X:\",train_X.shape)\n",
    "print(\"train_Y:\",train_Y.shape)\n",
    "print(\"test_X: \",test_X.shape)\n",
    "print(\"test_Y: \",test_Y.shape)\n"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:14:27.532135700Z",
     "start_time" : "2023-11-07T11:14:27.203644900Z"
    }
   },
   "id": "fa1058e25594edde"
  },
  {
   "cell_type": "markdown",
   "source" : [
    "## 训练"
   ] ,
   "metadata" : {
    "collapsed": false
   },
   "id" : "e9daaa394997ceda"
  },
  {
   "cell_type": "code",
   "execution_count" : 44,
   "outputs" : [
    {
     "name": "stdout",
     "output_type" : "stream",
     "text" : [
      "Epoch 1/20\n",
      "1875/1875 [==============================] - 7s 3ms/step - loss: 0.8264 - accuracy: 0.9071 - val_loss: 0.1784 - val_accuracy: 0.9490\n",
      "Epoch 2/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1838 - accuracy: 0.9479 - val_loss: 0.1617 - val_accuracy: 0.9553\n",
      "Epoch 3/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1571 - accuracy: 0.9575 - val_loss: 0.1908 - val_accuracy: 0.9456\n",
      "Epoch 4/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1317 - accuracy: 0.9648 - val_loss: 0.1557 - val_accuracy: 0.9560\n",
      "Epoch 5/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1158 - accuracy: 0.9696 - val_loss: 0.1351 - val_accuracy: 0.9661\n",
      "Epoch 6/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1077 - accuracy: 0.9726 - val_loss: 0.2141 - val_accuracy: 0.9664\n",
      "Epoch 7/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.1007 - accuracy: 0.9758 - val_loss: 0.1491 - val_accuracy: 0.9679\n",
      "Epoch 8/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0893 - accuracy: 0.9784 - val_loss: 0.1305 - val_accuracy: 0.9681\n",
      "Epoch 9/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0814 - accuracy: 0.9805 - val_loss: 0.1303 - val_accuracy: 0.9717\n",
      "Epoch 10/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0754 - accuracy: 0.9825 - val_loss: 0.1510 - val_accuracy: 0.9728\n",
      "Epoch 11/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0729 - accuracy: 0.9839 - val_loss: 0.1702 - val_accuracy: 0.9696\n",
      "Epoch 12/20\n",
      "1875/1875 [==============================] - 7s 4ms/step - loss: 0.0684 - accuracy: 0.9848 - val_loss: 0.1207 - val_accuracy: 0.9768\n",
      "Epoch 13/20\n",
      "1875/1875 [==============================] - 10s 5ms/step - loss: 0.0629 - accuracy: 0.9860 - val_loss: 0.1732 - val_accuracy: 0.9748\n",
      "Epoch 14/20\n",
      "1875/1875 [==============================] - 11s 6ms/step - loss: 0.0566 - accuracy: 0.9873 - val_loss: 0.1805 - val_accuracy: 0.9678\n",
      "Epoch 15/20\n",
      "1875/1875 [==============================] - 12s 6ms/step - loss: 0.0674 - accuracy: 0.9858 - val_loss: 0.1268 - val_accuracy: 0.9764\n",
      "Epoch 16/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0556 - accuracy: 0.9884 - val_loss: 0.1818 - val_accuracy: 0.9749\n",
      "Epoch 17/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0620 - accuracy: 0.9873 - val_loss: 0.1315 - val_accuracy: 0.9796\n",
      "Epoch 18/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0500 - accuracy: 0.9893 - val_loss: 0.1497 - val_accuracy: 0.9770\n",
      "Epoch 19/20\n",
      "1875/1875 [==============================] - 5s 3ms/step - loss: 0.0615 - accuracy: 0.9879 - val_loss: 0.1698 - val_accuracy: 0.9772\n",
      "Epoch 20/20\n",
      "1875/1875 [==============================] - 6s 3ms/step - loss: 0.0564 - accuracy: 0.9888 - val_loss: 0.1915 - val_accuracy: 0.9775\n",
      "313/313 [==============================] - 1s 3ms/step - loss: 0.1915 - accuracy: 0.9775\n",
      "loss: [0.1914634257555008, 0.9775000214576721]\n"
     ]
    }
   ],
   "source": [
    "epochs = 20\n",
    "model.compile(\n",
    "    loss = 'sparse_categorical_crossentropy',\n",
    "    optimizer = 'adam',\n",
    "    metrics = ['accuracy']\n",
    ")\n",
    "fit = model.fit(train_X, train_Y,\n",
    "                validation_data = (test_X,test_Y),\n",
    "                epochs = epochs)\n",
    "\n",
    "print(\"loss:\", model.evaluate(test_X, test_Y))"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:16:44.590790600Z",
     "start_time" : "2023-11-07T11:14:27.532135700Z"
    }
   },
   "id": "7bb7dcd8391ad0a"
  },
  {
   "cell_type": "code",
   "execution_count" : 45,
   "outputs" : [
    {
     "data": {
      "text/plain": "<Figure size 640x480 with 1 Axes>",
      "image/png" : "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RER+hwGHiIiI/A4DDhEREfkdBhwiIiLyOww4RERE5HcYcIiIiMjvMOAQERGR32HAISIiIr/DgENERER+hwGHiIiI/A4DDhEREfkdBhwiIiLyOww4RERE5HcYcIiIiMjvMOAQERGR32HAISIiIr/DgENERER+hwGHiIiI/A4DDhEREfkdBhw3EksKYV78GPIemOntohAREXVpCm8XwK8EBQM5F2EGIDcZAYXS2yUiIiLqktiC406qUEAZIN3Wlnq3LERERF0YA44bCYIAqCOlO9oS7xaGiIioC2PAcTd1FABALGMLDhERkbcw4LiZwBYcIiIir/OJQcYbNmzA2rVrodVqkZycjDlz5iA9Pb3J/Xfu3Im1a9ciLy8PKpUKgwcPxt13342wsLAOLHUTagOOWF4KwctFISIi6qq83oKzZ88erFixArfccguWLFmC9PR0vPHGGyguLna6/6+//oq//e1vmDRpEpYuXYqnnnoK586dw4cfftjBJXdOqO2iQhlbcIiIiLzF6wFn3bp1mDx5MqZMmWJvvYmOjsbGjRud7n/69GnExsbi+uuvR2xsLPr164err74a58+f7+CSN8HWgsNZVERERF7j1YBjNptx/vx5DBo0yGF7RkYGTp065fQ5ffv2RUlJCQ4fPgxRFKHVarF3714MGTKkI4rcIkFT24LDMThERERe49UxOBUVFbBarYiIiHDYHhERAa1W6/Q5ffv2xR/+8Ae8++67MJlMsFgsGD58OO69994mX8dkMsFkMtnvC4KA4OBg+223snVR1bbguP34PsRWN3+uY31dqb6sq//qSvVlXbs2nxhk7OyENHWSsrOz8cknn+C2227DoEGDUFZWhn/961/46KOP8PDDDzt9zpo1a/DVV1/Z76empmLJkiWIiYlxTwXqsWo0yAEAYw3iw8MgC/WBgc8eFh8f7+0idKiuVF/W1X91pfqyrl2TVwNOeHg4ZDJZo9aa8vLyRq06NmvWrEHfvn0xY8YMAED37t0RFBSERYsW4Y477oBGo2n0nJkzZ2L69On2+7bwVFRUBLPZ7Kba1B1bCAmDqK9E/qlMCInd3Hp8XyIIAuLj45Gfnw9RFL1dHI/rSvVlXf1XV6ov6+p/FApFqxsnvBpwFAoFevbsiePHj2PkyJH27cePH8eIESOcPqempgZyudxhm0wmDSVq6qQqlUoolc6vC+WJN4I8KgZmfSXEshIgIcXtx/c1oij69S9UQ12pvqyr/+pK9WVduyavz6KaPn06Nm/ejC1btiA7OxsrVqxAcXExpk6dCgBYtWoV/va3v9n3Hz58OPbv34+NGzeioKAAv/76Kz755BP06tULkZGR3qqGA3mUlC5FDjQmIiLyCq+PwRk7diwqKyvx9ddfo6ysDCkpKVi4cKG9CaqsrMxhTZyJEyeiuroa69evx6effoqQkBAMGDAAd911l7eq0Ig8srb5jFPFiYiIvMLrAQcArr32Wlx77bVOH5s3b16jbdOmTcO0adM8XSyX2VpwOFWciIjIO7zeReWP6rqo2IJDRETkDQw4HsAuKiIiIu9iwPGAui4qBhwiIiJvYMDxAHvAKS+FaLV6tzBERERdEAOOB8g1UYAgAFYrUFnu7eIQERF1OQw4HiDIFUC4WrrDbioiIqIOx4DjKWpeVZyIiMhbGHA8RFBLqypzqjgREVHHY8DxlNqAwxYcIiKijseA4yGCvYuKLThEREQdjQHHU9hFRURE5DUMOJ6i4SBjIiIib2HA8RAhwjYGhy04REREHY0Bx1NsLTi6Cogmk3fLQkRE1MUw4HhKSBigUEi3y9mKQ0RE1JEYcDxEEAQgglPFiYiIvIEBx5M0nCpORETkDQw4HmQbaMyp4kRERB2LAceTOFWciIjIKxhwPEnNqeJERETewIDjSbWXa2AXFRERUcdiwPEggS04REREXsGA40n1pomLoujdshAREXUhDDieZGvBqTEAhmrvloWIiKgLYcDxICEoGAhWSXfYTUVERNRhGHA8Tc2p4kRERB2NAcfT1Fzsj4iIqKMx4HhY3UwqtuAQERF1FAYcT+NUcSIiog7HgONpEVzsj4iIqKMx4HiYoGEXFRERUUdjwPG0CHZRERERdTQGHE+zXVG8vBSi1erdshAREXURDDieFq4BBAGwWABdhbdLQ0RE1CUw4HiYoFAAYRHSHY7DISIi6hAMOB2BU8WJiIg6FANOR4jgasZEREQdiQGnAwgaXo+KiIioI7kccD7++GPk5ua6syz+i1PFiYiIOpTC1Sdu374dGzZswMCBA3Hddddh+PDhEATBnWXzHxquZkxERNSRXA44f//737F9+3Zs3LgRb7/9NqKionDNNddg8uTJCA8Pd2cZOz1BHQkRYBcVERFRB3E54AQFBeHaa6/FtddeixMnTmD9+vX48ssvsXr1aowdOxbXXXcd0tLS3FnWzkttG4PDFhwiIqKO4HLAqW/gwIEYOHAgSkpKsGzZMuzYsQM7duxAWloabrnlFgwfPtwdL9N52aaJV5ZDNJsgKJTeLQ8REZGfc8ssKqPRiM2bN2PJkiU4efIkkpOTMWvWLFitVrz11lv46quv3PEynVdIGCCvzZLlWq8WhYiIqCtoVwtOfn4+NmzYgG3btqG6uhqDBw/GXXfdhYyMDADAbbfdhlWrVmH9+vW47bbb3FLgzkiQyaRWnJJCaRxOVIy3i0REROTXXA44b7zxBo4fP47AwEBMmjQJ1113HeLj4xvtN3z4cPz3v/9tVyH9gj3gcBwOERGRp7kccAoKCnDPPfdg0qRJCAoKanK/lJQULF682NWX8R9q22rGJeBkeiIiIs9yOeC89957rdovODgY/fv3d/Vl/IagjqqdKs4WHCIiIk9zeZBxbm4uMjMznT6WmZmJvLw8lwvll+wX3ORaOERERJ7mcsBZuXIlDhw44PSxgwcP4tNPP3W5UH5JzQtuEhERdRSXu6jOnz+PKVOmOH2sf//+2LVrl8uF8kdCRCS7qIiIPKimpgY1NTUO26qrq2E0Gr1Uoo7lL3UVBAGhoaHtvvyTywGnqqqqycHFAQEB0Ov1LhfKL/GK4kREHqPX6yEIAsLCwhw+GJVKJUwmkxdL1nH8pa5GoxE6nQ5hYWHtOo7LXVSRkZE4e/as08fOnj0LtVrt6qH9k20MjqEaoqHKu2UhIvIzZrMZKpWKF332AwEBARBFsd3HcTngjBgxAv/9739x4sQJh+0nT57Ef//7X4wcObLdhfMnQpAKCAqW7rCbiojIrRhsqCGXu6huu+02HDt2DK+++ioSExMRGRmJ0tJS5Obm2i/VQA2oI4H8HCngxCd7uzRERER+y+WAo1Kp8Prrr2PdunU4duwYiouLER4ejtmzZ+OGG25odvG/LksdBeTncLE/IiIiD2vXtaiCgoJw2223denrTLWFoOZMKiIi8oxhw4bhvvvuwwMPPNDuY+3ZswezZs1CZmYmIiIi3FC6jteugENtFGFb7I8Bh4iIpOEe/fv3xyuvvNLuY23YsAEBAQFuKJV/aFfAycvLw//+9z/k5OQ0mnsvCAIWLVrUrsL5ndqp4iKnihMRUSuIogiLxQKFouWP6+joaL+YJu4uLs+iysrKwrPPPotDhw7h6NGj0Ov1yM/PR2ZmJgoKCtwyxcvfCGq24BARdQRRFCHWGLzz1crPvyeeeAI//fQTli9fjqSkJCQlJeHLL79EUlIStm3bhmnTpiE1NRX79u3DxYsXMXfuXAwaNAi9e/fG9ddfjx07djgcb9iwYfjoo4/s95OSkrBq1Srcd999SEtLw7hx47Bx40aXf6bff/89Jk2ahNTUVIwaNQoffvihw+MrVqzAuHHj0LNnTwwaNMihq2zdunWYMmUK0tLSMGDAANx+++2oqvLskikut+B8/vnnGDRoEJ588knceeedeOihh9CzZ08cPnwYH3zwAe644w53ltM/qG2L/THgEBF5lLEG1kdno6blPd1O9rf/AIEtT7R55ZVXcP78efTr1w/PPPMMAODUqVMAgNdeew2LFi1Ct27dEB4ejry8PEyePBnPPvssAgMDsXr1asydOxc7duxAUlJSk6+xdOlSvPDCC3jhhRfwySef4NFHH8W+ffug0WjaVKfjx4/joYcewlNPPYUZM2bg4MGD+OMf/wiNRoPbb78dx44dw6JFi/CXv/wFw4cPh1arxb59+wAABQUFmDdvHp5//nlMmzYNOp0O+/bt83hDiMsB58KFC7j//vvtaw/YCjp06FDceOONWLVqFV5++eVWHWvDhg1Yu3YttFotkpOTMWfOHKSnpze5v8lkwldffYWdO3dCq9UiKioKM2fOxOTJk12tTseo14IjiiLXbSAi6sLCw8MREBCAoKAgxMbGAoB9Ad358+fjqquusu8bGRmJAQMG2O8vWLAA69evx8aNGzF37twmX2P27Nm4+eabAQDPPfccPv74Yxw9ehSTJk1qU1n/8Y9/YPz48XjyyScBAGlpaThz5gw+/PBD3H777cjJyYFKpcLVV1+N0NBQJCcnY+DAgQCAwsJCmM1mXH/99UhOlpZIae4z3l1cDjh6vR6hoaGQyWSQy+UOl2bo2bMnvvrqq1YdZ8+ePVixYgXuv/9+9O3bF5s2bcIbb7yBd955B9HR0U6f884776C8vBwPPfQQ4uPjUVFRAYvF4mpVOk5EbWK2mAFdBRDWOUemExH5vIBAyP72H+9cviAgsN2HyMjIcLhfVVWFpUuXYtOmTSgoKIDZbIbBYEBOTk6zx6kfJFQqFUJDQ1FcXNzm8pw5cwbXXnutw7YRI0bgn//8JywWC6666iokJydjzJgxmDhxIiZNmoRp06YhODgY/fv3x/jx4zFlyhRMmDABEyZMwA033ODxKx6061INFRUVAID4+HhkZmbaH8vKymr1Ojjr1q3D5MmTMWXKFHvrTXR0dJP9hEePHkVmZiYWLlyIjIwMxMbGolevXujbt6+rVekwgkJZF2rYTUVE5DGCIEAIDPLOlxta51UqlcP9V199FT/88AMWLFiAb775Bhs3bkS/fv1avLimUqls9HOxWq1tLo+zXof6XUyhoaFYv349li1bhri4OLz99tu4+uqrUV5eDrlcji+++AL/+te/0KdPH3zyySe46qqrkJWV1eZytIXLLTh9+/bF6dOnMXLkSIwfPx6rV6+GVquFQqHAtm3bcOWVV7Z4DLPZjPPnz9ubz2wyMjLs/ZANHTx4EGlpafjvf/+LHTt2ICgoCMOGDcMdd9zR5PQ4k8nkkOAFQUBwcLD9tjvZjtfkcdWRQGU5UF4KoVtPt752R2uxrn6mK9WXdfVfXa2+vk6pVLYqcOzfvx+zZs3CtGnTAEi9KNnZ2Z4unl2fPn2wf/9+h20HDx5Ez549IZfLAQAKhQJXXXUVrrrqKjz11FNIT0/H7t27cf3110MQBIwYMQIjRozAk08+iZEjR+LHH3/E73//+yZf02tXE7/llltQVlYGALj55puh1Wqxa9cuCIKAMWPG4O67727xGBUVFbBarY0WEYqIiIBWq3X6nIKCAvz6669QKpWYP38+KioqsHz5cuh0OjzyyCNOn7NmzRqHLrPU1FQsWbIEMTExraxt28XHxzvdXhSXCMPlC4gQLQhNSPDY63ekpurqr7pSfVlX/+Vv9a2urm7UWmHT1HZf0L17dxw9ehR5eXkICQmBTCZ1rCiVSodyp6amYv369Zg2bRoEQcCSJUtgtVohl8sd9mvpPiAFkZZ+JrZp6bZyzJs3D9dccw3+8pe/4Oabb8aBAwewYsUKLFmyBEqlEhs3bsSlS5cwevRoqNVqbNq0CVarFX379sXx48exc+dOTJw4EdHR0Th8+DBKS0uRnp7eZDkCAgKQ0M7PSJcDTnR0NOLi4gAAMpkM9957L+69916XjuUspTWV3GxNYn/4wx/sTXgmkwlLly7F/fff77QVZ+bMmZg+fXqjYxcVFcFsNrtU5qYIgoD4+Hjk5+c7HSFuCQ4BAJRfOo/KvDy3vnZHa6mu/qYr1Zd19V/+Wl+j0eh0rI1XxuC0wQMPPIAnnngC48ePh8FgwNKlSwE07nlYvHgxnnrqKdxwww2IjIzEvHnz7ONP6+/X0n1A6j1p6Wdi+2y0lSM9PR0ffvgh3n77bSxduhSxsbF45plncOutt8JkMiEkJATr1q3DW2+9BYPBgNTUVCxbtsw+GHnPnj34+9//Dp1Oh6SkJCxatAhXXXVVk+UwGo3Ic/IZqVAoWt044VLAMRqNuPvuu/H000+366rh4eHhkMlkjVprysvLm1waWq1WIzIy0qF/MikpCaIooqSkxGnia5iE6/PUL7gois6PXTuTSiwr8Zs/Lk3W1U91pfqyrv6rq9XXV6WlpeG7775z2Hb77bc32i8lJQWrV6922DZnzhyH+4cOHXIIDM4GIP/yyy+tKtfYsWMbPf+GG27ADTfc4HT/kSNHNjm5qHfv3vj3v//dqtetr73vT5cGGQcEBCAsLAyBge0bKa5QKNCzZ08cP37cYfvx48ebHDTcr18/lJWVwWAw2Lfl5eVBEARERUW1qzwdwhZwOMiYiIjIY1yeRTVs2LBGA45cMX36dGzevBlbtmxBdnY2VqxYgeLiYkydOhUAsGrVKvztb3+z7z9+/HiEhYXh/fffR3Z2NjIzM/Gvf/0LkyZN6hTX4BDsi/3xcg1EROQdCxYsQO/evZ1+LViwwNvFcwuXx+CMGzcOH3zwAd5//32MGjXK6aqIPXu2PEto7NixqKysxNdff42ysjKkpKRg4cKF9j62srIyhzn7QUFBeOGFF/Dxxx/jueeeQ1hYGMaMGdN5Vk7m5RqIiMjL5s+fj4ceesjpY2FhYR1cGs8QRBc7uZz1ETb05ZdfunLoDlNUVOT2wWeCICAhIQF5eXlO+w/FijJYn74HEATI3v8aQisuoOarWqqrv+lK9WVd/Ze/1reiogLh4eGNtvv6IGN38qe6Nnc+PTrIGAAefvhhV5/atYVGAHI5YLEAFWVApOemqhMREXVVLgeciRMnurEYXYcgk0mXbCgtlrqpGHCIiIjczuVBxtQOHGhMRETkUS634Lz//vvNPi4IAruxmlJvqjgXSyciInI/lwPOyZMnG23T6XQwGAxQqVQICQlpV8H8maCOgghwJhUREZGHuBxwli1b5nT7iRMn8M9//hNPPfWUy4Xye/ap4uyiIiIi140aNQr3338/HnjggRb3TUpKwvLly3Hdddd1QMm8z+1jcAYOHIjrrrsOn3zyibsP7T8iuJoxERGRJ3lkkHFycjLOnj3riUP7BUFjG2TMgENEROQJHgk4mZmZThfooVpczZiIyKNEUYTBbIXBZJW+d+BXaxdQ/OyzzzBs2DBYrVaH7XPmzMHjjz+OixcvYu7cuRg0aBB69+6N66+/Hjt27HDbz+iXX37BrFmzkJaWhgEDBuDZZ5+FXq+3P75nzx7ccMMN6NWrF9LT03HTTTchOzsbgDQO97bbbkOfPn3Qt29fXHfddTh27JjbyuYOLo/BcXbVUJPJhEuXLuHo0aOYMWNGuwrm12zTxKv1EGsMEAKDvFseIiI/U2MRcfuXp73y2l/e3gdBipbnyE6fPh2LFi3C7t27ceWVVwIAtFottm/fjhUrVkCv12Py5Ml49tlnERgYiNWrV2Pu3LnYsWMHkpKS2lXG6upq3HXXXRg6dCi+//57FBcXY/78+Xj++efx7rvvwmw247777sOdd96JZcuWwWQy4ciRIxAEqV6PPfYYBgwYgD/96U+QyWQ4efIkFD62Mr/LpWl42XZAujp4bGwsZs+ezYDTnKBgIDAIqDFIrThxid4uERERdTCNRoOJEyfi22+/tQecdevWQa1WY/z48ZDL5RgwYIB9/wULFmD9+vXYuHEj5s6d267X/uabb2AwGPDee+9BpVIBAF577TXMmTMHzz//PBQKBSoqKnD11VejR48eAIDevXvbn5+Tk4OHHnoIvXr1AtC6a092NJcDjq9fZ8qXCYIgteIU5DDgEBF5QKBcwJe394FSoYTJ3LHXZwqUt36Fs5kzZ2LBggV44403EBgYiDVr1mDGjBmQy+WoqqrC0qVLsWnTJhQUFMBsNsNgMCAnJ6fdZTxz5gzS09Pt4QYARowYAavVinPnzmH06NGYPXs2fvvb3+LKK6/ElVdeiRtvvBFxcXEAgAcffBDz58/H119/jSuvvBLTp0+3ByFfwZWMvcW+2B+nihMRuZsgCAhSyBCklEnfO/DL1o3TGlOnToXVasXmzZuRk5ODffv24dZbbwUAvPrqq/jhhx+wYMECfPPNN9i4cSP69esHo9HY7p+PKIpNltO2/Z133sHatWsxfPhwrF27FldeeSUOHToEAHj66aexZcsWTJkyBbt378akSZPw448/trtc7uRywDl06BDWr1/v9LH169fj8OHDLheqKxAiONCYiKirCw4OxrRp07BmzRr897//Rc+ePZGRkQEA2L9/P2bNmoVp06YhPT0dsbGx9kG+7dWnTx9kZmaiqqrKvu3AgQOQyWQO3U0DBw7EY489hrVr16Jv37749ttv7Y+lpaXhwQcfxOeff45p06b5XM+OywHH1n/nTE1NDdasWeNyoboEDRf7IyIi4JZbbsHmzZvxxRdf4JZbbrFv79GjB3788UecOHECJ0+exLx58xrNuGrPawYGBuLxxx/Hr7/+it27d+PFF1/ErbfeipiYGGRlZeHNN9/EwYMHkZ2dje3bt+P8+fPo1asXqqur8fzzz2PPnj3Izs7GgQMHcOzYMYcxOr7A5TE4ubm5SE1NdfpYamqqQ8ojJzhVnIiIAIwbNw5qtRrnzp3DzJkz7dtfeuklPPXUU7jpppsQGRmJefPmQafTueU1g4OD8e9//xuLFi3CDTfcgKCgINxwww1YvHix/fGzZ89i9erVKCsrQ2xsLObOnYu7774bZrMZZWVlePzxx1FcXIzIyEhMmzYNTz/9tFvK5i4uBxyTyQSz2dzkY+7oI/RntutRcQwOEVHXJpfLnQ7rSElJaTRjec6cOQ739+3b1+rXaTg4OT093emMaACIiYnB8uXLnT4WEBDQ4gW3fYHLXVSJiYn2wUYNHTp0CImJnBnULLbgEBEReYzLLTiTJk3CypUroVarcc0110CtVkOr1WLjxo3YsmULfve737mznP5HXXe5huZGsxMREbXkm2++wYIFC5w+lpycjK1bt3ZwibzP5YBz3XXX4dy5c/j666/x9ddfQyaT2Qc/XXnllbj++uvdVki/ZJtFZTYB+koglJe2ICIi11xzzTUYMWKE06EjSqXSCyXyPpcDjiAIePTRRzFlyhQcPXoUFRUVCA8Px5AhQ9CvXz93ltEvCUolEBoG6CqlbioGHCIiclFoaCg0Gg1Mpo5d1NCXtfvCEenp6UhPT3dHWboedVRtwCkBknt4uzRERER+w+VBxqdPn8aePXucPrZnzx6cOXPG5UJ1GfbVjDnQmIiIyJ1cDjiff/45srKynD6WnZ2NL774wuVCdRWCfaAxp4oTERG5k8sBJysrC3369HH6WO/evXHp0iWXC9VlcKo4ERGRR7gccAwGA2Qy508XBAHV1dUuF6rLqG3BYRcVERGRe7kccGJjY3Hy5Emnj508eRIxMTEuF6qrENiCQ0REbjJs2DB89NFH3i6Gz3A54IwbNw7ff/99o8WDtm3bhh9++AHjxo1rd+H8HgMOEVGXdtttt2HRokVuOdaGDRtw1113ueVY/sDlaeI333wzTp48iQ8//BAff/wxNBoNysrKYDQaMWDAAIcLhlETbIOMK7QQLRYIcrl3y0NERD5FFEVYLBYoFC1/XEdHR3MdnHpcbsFRKBR48cUXMW/ePIwYMQKxsbEYMWIE5s2bhxdeeKFVJ6PLCwsHZDJAtAIVWm+XhoiIOtATTzyBn376CcuXL0dSUhKSkpLw5ZdfIikpCdu2bcO0adOQmpqKffv24eLFi5g7dy4GDRqE3r174/rrr8eOHTscjtewiyopKQmrVq3Cfffdh7S0NIwbNw4bN25sVdksFguefvppjB49Gmlpabjyyivxz3/+s9F+X3zxBSZNmoTU1FQMGTIEzz//vP2x8vJyPPvssxg0aBB69uyJyZMn43//+5+LP622a1cKkclkuOqqq3DVVVc5bLdardi/fz9GjhzZrsL5O0Emly7ZUFYsTRXXRHm7SEREfkFq+QAEQYTZLHboa8vlaNX1BV955RWcP38e/fr1wzPPPAMAOHXqFADgtddew6JFi9CtWzeEh4cjLy8PkydPxrPPPovAwECsXr0ac+fOxY4dO5CUlNTkayxduhQvvPACXnjhBXzyySd49NFHsW/fPmg0mmbLZrVakZCQgA8//BCRkZE4ePAgnn32WcTGxmLGjBkAgJUrV+KVV17BwoULMWnSJFRWVuLAgQP25991113Q6/X461//iu7du+P06dOQd2BPhVubWXJycrB161Zs374dFRUV+PLLL915eP+ktgUcjsMhInIXiwX48etyr7z2tFsj0JpOjPDwcAQEBCAoKAixsbEAgLNnzwIA5s+f79B4EBkZiQEDBtjvL1iwAOvXr8fGjRsxd+7cJl9j9uzZuPnmmwEAzz33HD7++GMcPXoUkyZNarZsSqXSHroAoFu3bjh48CC+++47e8D5y1/+ggcffBD333+/fb/BgwcDAHbu3ImjR49i27ZtSEtLAwB07969pR+JW7U74BgMBuzZswdbt27F6dOnAQCpqam4/fbb2124LqHeasa8njgREQFARkaGw/2qqiosXboUmzZtQkFBAcxmMwwG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6zaiKGLlypXo168funXr1uR+Go0GDz74IHr27Amz2YwdO3bg1VdfxeLFi1tsHfC23r17Y968eUhMTIRWq8U333yDF154AUuXLkVYWFij/f3hvALA/v37odfrMXHixCb36czntT7b76iz81ZcXNzs8xr+16tWq5v8nfdVRqMRq1atwrhx45oNOImJiXjkkUfQrVs3VFdX44cffsCLL76It956CwkJCR1Y4rYZMmQIxowZg+joaBQWFuLLL7/EK6+8gj/96U9Nru7rD+d2+/btCAoKanHcUGc9r+3BgOMCZ//pNvffb8PHbItHt/Qfs69Yvnw5srKy8MorrzS7X2JiosMg5D59+qC4uBjfffedz38Q1m8O7tatG/r06YPHHnsM27dvx/Tp050+p7OfVwDYunUrBg8e3Oyg0858Xp1p6ry1hSiKneo8m81mvPvuuxBF0elA8vr69OmDPn362O/37dsXCxYswI8//oh7773X00V12dixY+23u3XrhrS0NDzyyCM4fPhwi2NT6uts53br1q248sorWxxL01nPa3twmngbhIeHQyaTNUr35eXlTTaDOvtvoKKiAnK5vFXdPd728ccf49ChQ1i8eDGioqLa/Pw+ffogPz/fAyXzrKCgIHTr1g15eXlOH+/s5xUAioqKcPz4cft4srbojOfV9p+6s/PWXDeGs3Pd3O+8rzGbzXjnnXdQVFSEF154ocXuqYZkMhnS0tI63fnWaDSIiYlp8ncY6Pzn9pdffkFubi4mT57c5ud21vPaFgw4baBQKNCzZ08cP37cYfvx48fRt29fp8/p3bt3o/2PHTuGnj17+vQ4DVEUsXz5cuzbtw+LFi1CbGysS8e5cOFCpxzUZjKZkJOT02RXU2c9r/Vt3boVERERGDp0aJuf2xnPa2xsLNRqtcN5M5vNyMzMbPL3F5DC3M8//+yw7fjx4w7/DfsqW7jJz8/Hiy++6LS7tSWiKOLSpUud7nxXVlaipKSk2e7iznxuAWDLli3o2bMnevTo0ebndtbz2hYMOG00ffp0bN68GVu2bEF2djZWrFiB4uJi+wDFVatW4W9/+5t9/2uuuQbFxcX29VK2bNmCLVu24MYbb/RWFVpl+fLl2LlzJx5//HEEBwdDq9VCq9XCaDTa92lY1++//x779+9HXl4eLl++jFWrVmHfvn247rrrvFGFNvn000+R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"
     },
     "metadata" : {},
     "output_type" : "display_data"
    }
   ] ,
   "source": [
    "pyplot.style.use(\"ggplot\")\n",
    "pyplot.figure()\n",
    "pyplot.plot(range(epochs), fit.history[\"loss\"], label=\"train_loss\")\n",
    "pyplot.plot(range(epochs), fit.history[\"val_loss\"], label=\"val_loss\")\n",
    "pyplot.plot(range(epochs), fit.history[\"accuracy\"], label=\"train_acc\")\n",
    "pyplot.plot(range(epochs), fit.history[\"val_accuracy\"], label=\"val_acc\")\n",
    "pyplot.title(\"title\")\n",
    "pyplot.xlabel(\"Epoch #\")\n",
    "pyplot.ylabel(\"Loss/Accuracy\")\n",
    "pyplot.legend()\n",
    "pyplot.show()"
   ],
   "metadata": {
    "collapsed": false,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:16:44.660325900Z",
     "start_time" : "2023-11-07T11:16:44.584430100Z"
    }
   },
   "id": "9295e475b215cc95"
  },
  {
   "cell_type": "code",
   "execution_count" : 62,
   "outputs" : [
    {
     "name": "stdout",
     "output_type" : "stream",
     "text" : [
      "INFO:tensorflow:Assets written to: MLP-MNIST/assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type" : "stream",
     "text" : [
      "INFO:tensorflow:Assets written to: MLP-MNIST/assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type" : "stream",
     "text" : [
      "4\n",
      "(28, 28)\n",
      "1/1 [==============================] - 0s 20ms/step\n",
      "predict [[6.3848029e-26 1.8576654e-11 9.1197871e-10 5.5588011e-33 9.9998713e-01\n",
      "  2.2778484e-11 1.2607207e-05 8.0123747e-10 1.3219515e-14 2.6355207e-07]]\n",
      "1/1 [==============================] - 0s 44ms/step\n",
      "predict [[6.3848029e-26 1.8576654e-11 9.1197871e-10 5.5588011e-33 9.9998713e-01\n",
      "  2.2778484e-11 1.2607207e-05 8.0123747e-10 1.3219515e-14 2.6355207e-07]]\n"
     ]
    },
    {
     "data": {
      "text/plain": "<Figure size 640x480 with 1 Axes>",
      "image/png" : "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"
     },
     "metadata" : {},
     "output_type" : "display_data"
    }
   ],
   "source": [
    "model.save(\"MLP-MNIST\")\n",
    "model1: Model = load_model(\"MLP-MNIST\")\n",
    "print(test_Y[6])\n",
    "print(test_X[6].shape)\n",
    "pyplot.imshow(test_X[6])\n",
    "print(\"predict\",model.predict(test_X[6].reshape((1,28,28))))\n",
    "print(\"predict\",model1.predict(test_X[6].reshape((1,28,28))))\n"
   ] ,
   "metadata": {
    "collapsed": false,
    "ExecuteTime" : {
     "end_time": "2023-11-07T11:37:10.255927100Z",
     "start_time" : "2023-11-07T11:37:06.651262700Z"
    }
   },
   "id": "initial_id"
  }
    ],
        "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
                "language" : "python",
                "name" : "python3"
        },
            "language_info" : {
            "codemirror_mode": {
                "name": "ipython",
                    "version" : 2
            },
                "file_extension" : ".py",
                "mimetype" : "text/x-python",
                "name" : "python",
                "nbconvert_exporter" : "python",
                "pygments_lexer" : "ipython2",
                "version" : "2.7.6"
        }
    },
        "nbformat": 4,
        "nbformat_minor" : 5
}