root = "/raid/workspace/MOT/MOT20/images/train/MOT20-01/gt/gt.txt"
COLUMN_MEANING = ['Frame number', 
                'Id number',
                'bbox left',
                'bbox top',
                'bbox width',
                'bbox height',
                'confidence score',
                'class',
                'visibility']

cnt = 0
with open(root, 'r') as f:
    for line in f: 
        line_lst = line.strip().split(',')
        print(line_lst)
        if line_lst[0] == '2' and line_lst[7] == '1':
            cnt += 1
    
print(cnt)