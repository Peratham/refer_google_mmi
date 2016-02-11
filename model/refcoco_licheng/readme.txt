# 1. baseline model with global context
th train.lua -ranking_weight 0 id 0

# 2. baseline model without global context
th train.lua -jemb_use_global 0 -ranking_weight 0 id 1

# 3. MMI model with global context
th train.lua id 10

# 4. MMI model without global context
th train.lua -jemb_use_global 0 -id 11

1's result:
testA: bleu1 0.415, bleu2 0.241, bleu3 0.138, rouge 0.372, meteor 0.153, IoU 61.85%
testB: bleu1 0.480, bleu2 0.286, bleu3 0.166, rouge 0.467, meteor 0.206, IoU 61.68%
val:   bleu1 0.458, bleu2 0.275, bleu3 0.162, rouge 0.422, meteor 0.182, IoU 

2's result:
testA: bleu1 0.422, bleu2 0.260, bleu3 0.161, rouge 0.376, meteor 0.156
testB: bleu1 0.476, bleu2 0.291, bleu3 0.184, rouge 0.469, meteor 0.205
val:   bleu1 0.466, bleu2 0.287, bleu3 0.178, rouge 0.427, meteor 0.184

3's result:
testA: bleu1 0.429, bleu2 0.264, bleu3 0.154, rouge 0.382, meteor 0.158
testB: bleu1 0.461, bleu2 0.281, bleu3 0.167, rouge 0.463, meteor 0.201
val:   bleu1 0.451, bleu2 0.275, bleu3 0.172, rouge 0.419, meteor 0.178

4's result:
testA: bleu1 0.422, bleu2 0.256, bleu3 0.156, rouge 0.381, meteor 0.158
testB: bleu1 0.490, bleu2 0.301, bleu3 0.186, rouge 0.472, meteor 0.209
val:   bleu1 0.466, bleu2 0.287, bleu3 0.186, rouge 0.429, meteor 0.184
