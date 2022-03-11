bazel build --config=opt -c opt //tensorflow/tools/pip_package:build_pip_package
rm -rf /tmp/tensorflow_pkg
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pushd /tmp/tensorflow_pkg
pip uninstall tensorflow -y
pkg=`ls .`
pip install $pkg --force-reinstall
python -c "import tensorflow as tf;print(tf.__git_version__)"
popd
