import torch
from metrics import MetronAtK

class Engine(object):
    def __init__(self):
        self._metron = MetronAtK(top_k=10)
        
    def evaluate(self, model,evaluate_data,dic_director,one_hot_vector, epoch_id):
        #Evaluate model
        model.eval()
        director_tp = []
        director_tn = []
        genre_tp=[]
        genre_tn=[]
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]
            
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            
            for i in test_items :
                director_tp.append(dic_director[i.item()])
            for j in negative_items :
                director_tn.append(dic_director[j.item()])    
                
            for i in test_items :
                genre_tp.append(one_hot_vector[i.item()])
                 
            for j in negative_items :
                genre_tn.append(one_hot_vector[j.item()])    
               
                
            director_tp = torch.LongTensor(director_tp)
            director_tp = director_tp.cuda()
            director_tn = torch.LongTensor(director_tn)
            director_tn = director_tn.cuda()
            genre_tp = torch.LongTensor(genre_tp)
            genre_tp = genre_tp.cuda()
            genre_tn = torch.LongTensor(genre_tn)
            genre_tn = genre_tn.cuda()    
            test_scores = model(test_users, test_items,director_tp,genre_tp)
            negative_scores = model(negative_users, negative_items,director_tn,genre_tn)
            
            #to cpu
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
            self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id+1, hit_ratio, ndcg))
        return hit_ratio, ndcg
