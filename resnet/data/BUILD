py_library(
    name = "get_dataset",
    srcs = ["get_dataset.py"],
    deps = ["//resnet/utils:logger", "//resnet/utils:batch_iter",
            ":imagenet", ":cifar"],
)

py_library(
    name = "cifar",
    srcs = ["cifar10.py", "cifar100.py", "cifar_input.py"],
    deps = ["//resnet/utils:logger"],
)

py_library(
    name = "imagenet",
    srcs = ["imagenet.py", "vgg_preprocessing.py", 
            "inception_preprocessing.py", "synset.py"],
    deps = ["//resnet/utils:logger"],
)

py_library(
    name = "data",
    srcs = ["__init__.py"],
    deps = [":get_dataset"],
    visibility = ["//visibility:public"],
)
