while getopts f option
do
case "${option}"
in
f) FORMAT=${OPTARG};;
esac
done

source activate enscalo_test
source /opt/intel/openvino/bin/setupvars.sh
python inference_model.py $FORMAT
