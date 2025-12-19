count = {}
already_seen_drugs = set()
with open('vitagraph.tsv', 'r') as fin:
  for line in fin.readlines()[1:]:
    line = line.strip().split('\t')
    head = line[0].split('::')
    tail = line[2].split('::')
    head_type = head[0]
    head_src, head_id = head[1].split(':')
    tail_type = tail[0]
    tail_src, tail_id = tail[1].split(':')
    if head_type == 'Compound' and head_id not in already_seen_drugs:
      already_seen_drugs.add(head_id)
      if head_src not in count:
        count[head_src] = 1
      else:
        count[head_src] += 1
    if tail_type == 'Compound' and tail_id not in already_seen_drugs:
      already_seen_drugs.add(tail_id)
      if tail_src not in count:
        count[tail_src] = 1
      else:
        count[tail_src] += 1

print(count)