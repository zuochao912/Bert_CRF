def read_corpus(filepath):
    sentences, tags = [], []
    sent, tag = [], []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            if len(line.strip()) !=1:
                if line == '\n':
                    if len(sent) > 1:
                        sentences.append(sent)
                        tags.append(tag)
                    sent, tag = [], []
                else:
                    line = line.strip().split(' ')
                    sent.append(line[0])
                    tag.append(line[1])
    return sentences, tags


def evaluate_ner(predict_tags, gold_tags):
    gold_entities, gold_entity = [], []
    predict_entities, predict_entity = [], []
    for line_no, sentence in enumerate(gold_tags):
        for char_no in range(len(sentence)):
            gold_tag_type = gold_tags[line_no][char_no]
            predict_tag_type = predict_tags[line_no][char_no]
            if gold_tag_type[0] == "B":
                gold_entity = [str(char_no) + "/" + gold_tag_type]
            elif gold_tag_type[0] == "I" and len(gold_entity) != 0 \
                    and gold_entity[-1].split("/")[1][1:] == gold_tag_type[1:]:
                gold_entity.append(str(char_no) + "/" + gold_tag_type)
            elif gold_tag_type[0] == "O" and len(gold_entity) != 0 :
                gold_entities.append(gold_entity)
                gold_entity=[]
            else:
                gold_entity=[]

            if predict_tag_type[0] == "B":
                predict_entity = [str(char_no) + "/" + predict_tag_type]
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 \
                    and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(str(char_no) + "/" + predict_tag_type)
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                predict_entities.append(predict_entity)
                predict_entity = []
            else:
                predict_entity = []
    acc_entities = [entity for entity in predict_entities if entity in gold_entities]
    acc_entities_length = len(acc_entities)
    predict_entities_length = len(predict_entities)
    gold_entities_length = len(gold_entities)
    # print(acc_entities[:10])
    print("acc_entities_length: ",acc_entities_length)
    print("predict_entities_length: ",predict_entities_length)
    print("gold_entities_length: ",gold_entities_length)

    if acc_entities_length > 0:
        step_acc = float(acc_entities_length / predict_entities_length)
        step_recall = float(acc_entities_length / gold_entities_length)
        f1_score = 2 * step_acc * step_recall / (step_acc + step_recall)
        return step_acc, step_recall, f1_score
    else:
        return 0, 0, 0


if __name__ == "__main__":
    #a=[["B-1","I-1","O","O","B-2","O","O"],["B-3","I-3","O","O","B-2","O","O"]]
    #b=[["B-1","I-1","O","O","B-2","O","O"],["B-2","I-2","O","O","B-2","O","O"]]
    #evaluate(a,b)
    sents, gold_tags = read_corpus("golden.txt")
    sents, predict_tags = read_corpus("predict.txt")
    acc, recall, f1 = evaluate_ner(predict_tags, gold_tags)
    print("The accuracy (f1-score) is: ", f1)






# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve


# def evaluate(y_pred, labels, preds=None, img_path=None):
#     TP, TN, FP, FN = 0, 0, 0, 0
#     for p, l in zip(y_pred, labels):
#         TP += (p == 1) & (l == 1)
#         TN += (p == 0) & (l == 0)
#         FN += (p == 0) & (l == 1)
#         FP += (p == 1) & (l == 0)
    
#     p = TP / (TP + FP)  
#     r = TP / (TP + FN)

#     print("="*50)
#     ALL = TP + FP + TN + FN
#     print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP/ALL, TN/ALL, FP/ALL, FN/ALL))
#     print("="*50)

#     acc = accuracy_score(y_pred, labels)
#     #roc_auc = roc_auc_score(y_pred, labels)
#     f1 = f1_score(y_pred, labels)
    
#     #if preds is not None:
#     #    precision, recall, thresholds = precision_recall_curve(labels, preds)
#     #    draw_p_r_curve(recall, precision, img_path)

#     return acc, p, r, f1  #, roc_auc


# def draw_p_r_curve(recall, precision, img_path):
#     plt.figure("P-R Curve")
#     plt.title('Precision/Recall Curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.plot(recall, precision)
#     plt.savefig('./checkpoints/imgs/{}.png'.format(img_path))