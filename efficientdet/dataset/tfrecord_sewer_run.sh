CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-1.균열-길이(Crack-Longitudinal,CL)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-1.균열-길이(Crack-Longitudinal,CL)/annotations/균열-길이(Crack-Longitudinal,CL).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-1.균열-길이(Crack-Longitudinal,CL)/annotations/균열-길이(Crack-Longitudinal,CL).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/CL" \
    --num_shards=50 

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-2.균열-원주(Crack-Circumferential,CC)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-2.균열-원주(Crack-Circumferential,CC)/annotations/균열-원주(Crack-Circumferential,CC).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-1.균열(Crack,CR)/1-1-2.균열-원주(Crack-Circumferential,CC)/annotations/균열-원주(Crack-Circumferential,CC).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/CC" \
    --num_shards=50 

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-2.표면손상(Surface-Damage,SD)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-2.표면손상(Surface-Damage,SD)/annotations/표면손상(Surface-Damage,SD).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-2.표면손상(Surface-Damage,SD)/annotations/표면손상(Surface-Damage,SD).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/SD" \
    --num_shards=50 

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-3.파손(Broken-Pipe,BK)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-3.파손(Broken-Pipe,BK)/annotations/파손(Broken-Pipe,BK).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-3.파손(Broken-Pipe,BK)/annotations/파손(Broken-Pipe,BK).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/BK" \
    --num_shards=50 

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-4.연결관-돌출(Lateral-Protruding,LP)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-4.연결관-돌출(Lateral-Protruding,LP)/annotations/연결관-돌출(Lateral-Protruding,LP).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-4.연결관-돌출(Lateral-Protruding,LP)/annotations/연결관-돌출(Lateral-Protruding,LP).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/LP" \
    --num_shards=50 

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-5.이음부-손상(Joint-Faulty,JF)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-5.이음부-손상(Joint-Faulty,JF)/annotations/이음부-손상(Joint-Faulty,JF).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-5.이음부-손상(Joint-Faulty,JF)/annotations/이음부-손상(Joint-Faulty,JF).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/JF" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-6.이음부-단차(Joint-Displaced,JD)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-6.이음부-단차(Joint-Displaced,JD)/annotations/이음부-단차(Joint-Displaced,JD).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-6.이음부-단차(Joint-Displaced,JD)/annotations/이음부-단차(Joint-Displaced,JD).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/JD" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-7.토사퇴적(Deposits-Silty,DS)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-7.토사퇴적(Deposits-Silty,DS)/annotations/토사퇴적(Deposits-Silty,DS).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-7.토사퇴적(Deposits-Silty,DS)/annotations/토사퇴적(Deposits-Silty,DS).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/DS" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-8.기타결함(Etc.,ETC)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-8.기타결함(Etc.,ETC)/annotations/기타결함(Etc.,ETC).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/1.손상/1-8.기타결함(Etc.,ETC)/annotations/기타결함(Etc.,ETC).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/ETC" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-1.이음부(Pipe-Joint,PJ)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-1.이음부(Pipe-Joint,PJ)/annotations/이음부(Pipe-Joint,PJ).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-1.이음부(Pipe-Joint,PJ)/annotations/이음부(Pipe-Joint,PJ).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/PJ" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-2.하수관로_내부(Inside,IN)/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-2.하수관로_내부(Inside,IN)/annotations/하수관로_내부(Inside,IN).json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-2.하수관로_내부(Inside,IN)/annotations/하수관로_내부(Inside,IN).json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/IN" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-1.하수관로_외부_맨홀/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-1.하수관로_외부_맨홀/annotations/하수관로_외부(Outside,OUT)_맨홀.json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-1.하수관로_외부_맨홀/annotations/하수관로_외부(Outside,OUT)_맨홀.json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/HL" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-2.하수관로_외부_인버트/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-2.하수관로_외부_인버트/annotations/하수관로_외부(Outside,OUT)_인버트.json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-2.하수관로_외부_인버트/annotations/하수관로_외부(Outside,OUT)_인버트.json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/INV" \
    --num_shards=50

CUDA_VISIBLE_DEVICES=3 python create_coco_tfrecord_pipe_defect.py --logtostderr \
    --image_dir="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-3.하수관로_외부_자동차/images/" \
    --image_info_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-3.하수관로_외부_자동차/annotations/하수관로_외부(Outside,OUT)_자동차.json" \
    --object_annotations_file="/hd/sewer/sewer_100/sewer_100_210215/2.비손상/2-3.하수관로_외부(Outside,OUT)/2-3-3.하수관로_외부_자동차/annotations/하수관로_외부(Outside,OUT)_자동차.json" \
    --output_file_prefix="/hd/sewer/sewer_100/tfrecord/CAR" \
    --num_shards=50
