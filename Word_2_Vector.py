#!/usr/bin/env python
# coding: utf-8

# GitHub: https://github.com/sagarsp1440/HumanAI

# In[ ]:


import numpy as np
from w2v_utils import *


# In[ ]:


words, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')


# In[ ]:


print("Example of words: ",list(words)[:10])
print("Vector for word 'person' = ", word_to_vec_map.get('person'))


# In[ ]:


def cosine_similarity(u, v):   

    distance = 0.0
  
    dot = np.dot(u,v)
    
    norm_u = np.sqrt(np.sum(np.square(u)))
    
    norm_v =  np.sqrt(np.sum(np.square(v)))
    
    cosine_similarity = dot / (norm_u*norm_v)
    
    
    
    return cosine_similarity


# In[ ]:


def most_similar_word(word, word_to_vec_map):

    
    word = word.lower()
    
    
     
    e = word_to_vec_map[word]
   
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100             
    best_word = None                   

    
    for w in words:        
        
        if w == word :
            continue
        
        
        cosine_sim = cosine_similarity(e, word_to_vec_map[w] )
        
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word, max_cosine_sim


# In[ ]:


brother = word_to_vec_map["brother"]
friend = word_to_vec_map["friend"]
computer = word_to_vec_map["computer"]
kid = word_to_vec_map["kid"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]


# In[ ]:


print("cosine_similarity(brother, friend) = ", cosine_similarity(brother, friend))
print("cosine_similarity(computer, kid) = ",cosine_similarity(computer, kid))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
print(most_similar_word('computer', word_to_vec_map))
print(most_similar_word('australia', word_to_vec_map))
print(most_similar_word('python', word_to_vec_map))


# In[ ]:


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
   
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
  
    words = word_to_vec_map.keys()
    max_cosine_sim = -100             
    best_word = None                  

    
    for w in words:        
        
        if w in [word_a, word_b, word_c] :
            continue
        
       
        cosine_sim = cosine_similarity((e_b-e_a),(word_to_vec_map[w]-e_c) )
        
       
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
      
        
    return best_word


# In[ ]:


triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))


# In[ ]:


triads_to_try = [('italy', 'italian', 'spain'), ('chinese', 'beijing', 'france'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]  
for triad in triads_to_try:  
print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))  


# In[ ]:


g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)


# In[ ]:


print ('List of names and their similarities with constructed vector:')

name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))


# In[ ]:


print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']


for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))


# In[ ]:


#The above codes result shows bias in the name based on the gender the name represents


# In[ ]:


def neutralize(word, g, word_to_vec_map):
    
    e = word_to_vec_map[word]
    
    e_biascomponent = (np.dot(e,g) / (np.linalg.norm(g) ** 2))*g
 
    e_debiased = e-e_biascomponent
    
    return e_debiased


# In[ ]:


e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))


# In[ ]:


def equalize(pair, bias_axis, word_to_vec_map):
    
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    
    mu = 0.5*(e_w1 + e_w2)

    
    mu_B = (np.dot(mu,bias_axis)/(np.linalg.norm(bias_axis)**2)) * bias_axis
    mu_orth = mu - mu_B


    e_w1B = (np.dot(e_w1,bias_axis)/(np.linalg.norm(bias_axis)**2)) * bias_axis
    e_w2B = (np.dot(e_w2,bias_axis)/(np.linalg.norm(bias_axis)**2)) * bias_axis
        
    
    corrected_e_w1B = (np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))/np.linalg.norm(e_w1-mu_orth-mu_B))*(e_w1B-mu_B)
    corrected_e_w2B = (np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))/np.linalg.norm(e_w2-mu_orth-mu_B))*(e_w2B-mu_B)

    
    e1 = corrected_e_w1B+mu_orth
    e2 = corrected_e_w2B+mu_orth
                                                                
    
    
    return e1, e2


# In[ ]:


print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))


# In[ ]:




