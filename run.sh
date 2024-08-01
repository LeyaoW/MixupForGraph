# node classification with mixup


LOG_FILE="logs/New_setting_1.log"
exec > >(tee ${LOG_FILE}) 2>&1
# python mixup.py --mixup True --dataset pubmed_raw --imb_class='01'
# python mixup.py --mixup True --dataset cora_raw --imb_class='01234'
echo "===arxiv==="
python mixup.py --mixup True --dataset Arxiv
#python mixup.py --mixup True --dataset History
echo "===photo==="
python mixup.py --mixup True --dataset Photo
echo "===computer==="
python mixup.py --mixup True --dataset Computer
