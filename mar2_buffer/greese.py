import networkx as nx
import time
def modu1(G,U,N,O):


    n=len(U);
    i=0
    m=0
    S=G.subgraph(U)
    cpt=float(S.number_of_edges())
   
    r=list(G.nodes())
    #print(r)
    while i< n:
        j=i
        #print(i)
        while j<n:
            
            
            b1=(O[r.index(U[i])])
           
         
            b2=(O[r.index(U[j])])
            if G.has_edge(U[i],U[j]):
                #temp=b1*b2*(1-((G.degree(U[i])*G.degree(U[j]))/(2*N)))
                temp=(1/(b1*b2))*(1-((G.degree(U[i])*G.degree(U[j]))/(2*N)))
                m=m+2*temp
            elif i==j:
                
                temp=-(1/(b1*b2))*(((G.degree(U[i])*G.degree(U[j]))/(2*N)))
                m=m+temp
            else:
                temp=-(1/(b1*b2))*((G.degree(U[i])*G.degree(U[j]))/(2*N))
                m=m+2*temp
            
            j=j+1
        i=i+1
    
    
    return(m)
def b(path,sep):
    
    #rint(len(ls))
    t=[]
    tri=[]
    G=nx.read_edgelist(path, comments='#', delimiter=sep, nodetype=int,encoding='utf-8')
    #G=nx.read_gml(path)
    print('aa')
    #print(G.nodes())
    #print(G.edges())
    k=0
    ii=0
    #N=34681189
    #N= 117619#196972
    #N=5485#613#254#159
    
    """G1=nx.Graph()
    r1=list(G.nodes())
    for k in r1:
        G1.add_node(r1.index(k))
    e1=list(G.edges())
    for kk in e1:
        G1.add_edge(r1.index(kk[0]),r1.index(kk[1]))
    for k in list(G1.edges()):
        fich.write(str(k[0]))
        fich.write(',')
        fich.write(str(k[1]))
        fich.write('\n')
    fich.close()
    """
    #i=0
    #while i<n:
        #G.add_node(i)
        #i=i+1
    
    #c= max(nx.connected_component_subgraphs(G), key=len)
    t=[]
    #=Gc
    ns=G.number_of_nodes()
    N=G.number_of_edges()
    print('aa')
    t=[]
    den=nx.density(G)
    etr=den*ns
    #print('diameter',nx.diameter(G))
    print('density',den)# equivalent moyenne des degree sd/ns
    if etr>7:
        th=0.7
        th1=2
    else:
        th=0.6
        th1=3
    
    th=0.5
    #print('average_clustering',nx.average_clustering(G))
    
    #print(len(G.edges()))
    
  




    ns=G.number_of_nodes()
    N=G.number_of_edges()
    w1=[]
    T1=G.nodes()
    #nodedeg=nx.degree(G,T1)
    tr=[]
    #mp=sorted(nodedeg.items(), key=itemgetter(1),reverse=True)
    #for k in mp:
    #    tr.append(k[0])
    #tr = [x for n, x in enumerate(mp) ]
    #print(tr)
    #bb= float(sum(nodedeg.values()))
    #bb=int(bb/len(T1))
    #print('bb',bb)
    #bb=bb/2
    tsp1= time.time()
    #i=5
    #tr=[]
    
    #print((tr))
    res=[]
    sup=[]
    tr=list(G.nodes())
    nf=[]
    np=[]
    rp=0
    b2=[]
    #for kk in tr
    kk=0
    #g2=G
    #print(tr)
    i1=0
    while kk<len(tr):
            c=[]
            i1=tr[kk]
            #rp=rp+1
        
            xx=list(G.neighbors(i1))
            #print(i1)
            T=[]
            #for i in xx:
                    #T.append([G.degree(i),i])
            #print('a')
            ii=0
            #while ii<len(xx):
            j=0
            mx=0
            
            while j<len(xx):
                o=len(sorted(nx.common_neighbors(G, i1, xx[j])))
                if o>mx:
                        mx=o
                        tp=j
                
                     
                j=j+1
            
            if mx>0:
                c.append(i1)
                c.append(xx[tp])
                
            
                #ne2=list(set().intersection(G.neighbors(c[1]),G.neighbors(c[0])))
                #T2=list(set().intersection(G.neighbors(c[1]),G.neighbors(c[0])))
                #c.extend(T2)
                #print(c)
                T=list(set().union(G.neighbors(c[0]))-set(c))
                #T=[]
                #print(T)
                
                
                
                #nodedeg=nx.degree(G,nei)
                #nodedeg1=nx.triangles(G,nei)
                
                for k in T:
                        rr=list(G.neighbors(k))
                        cpt=len(set(rr).intersection(c))
                        
                        n=len(c)
                        if (cpt/n)>=0.9 :
                            c.append(k)
                            T.remove(k) 
                        
                
                for k in T:
                        rr=list(G.neighbors(k))
                        cpt=len(set(rr).intersection(c))
                        
                        n=len(c)
                        if (cpt/n)>=0.8 :
                            c.append(k)
                            T.remove(k) 
                
                for k in T:
                        rr=list(G.neighbors(k))
                        cpt=len(set(rr).intersection(c))
                        
                        n=len(c)
                        if (cpt/n)>=th :
                            c.append(k)
                            T.remove(k) 
                for k in T:
                        rr=list(G.neighbors(k))
                        cpt=len(set(rr).intersection(c))
                        #n=len(c)
                        if  (cpt/len(rr))>0.5:
                            c.append(k)
                            #T.remove(k)                            
                #c=list(set(c))
                
                
                nn=len(tr)
                #print(c)
                tr=list(set(tr)-(set(c)))
                nn2=len(tr)
                n2=nn-nn2
                por=n2/len(c)
                
                if por>=0.5:
                    res.append(c)
                else:
                    i3=0
                    while i3<len(res):
                        rr=len(set(res[i3]).intersection(set(c)))
                        
                        if rr>=min(len(res[i3]),len(c))/2: 
                            res[i3].extend(c)
                            res[i3]=list(set(res[i3]))
                            #print(c)
                            #print(res[i3])
                            break
                        else:
                            i3=i3+1
              
                    if i3==len(res):
                        res.append(c)
                #print(rp)
                
                
                  

            else:
                
                kk=kk+1 
        
    
    
    
    sup=tr
    
    #ajout des sommets qui sont pas encore inclut dans le partitionement
    print('len',len(res))        
    res1=[]
    
    for k in sup:
            #print(k)
        
            max=0
            ne=list(G.neighbors(k))
            #print(ne)
            i=0
            while i <len(res):
                aa=len(set(ne).intersection(res[i]))
                
                if aa>=max:
                    max=aa
                    temp=i
                    if (aa)>len(ne)-aa:
                        #print('b')
                        i=len(res)
                i=i+1
            if max>0:
                res[temp].append(k)
            
    
    print('ff1')
    res.sort(key=len,reverse=True)
    
    
    r=0
    print('ff2')
    r=0
    print("r")
    while(r<len(res)):
        
            j=r+1
            while j<len(res):
                if  len(set(res[r]).intersection(res[j]))>=(len(res[j])/3):
                    res[r]=list(set(res[r]).union(res[j]))
                    res.pop(j)
                
                else:
                    j=j+1
            if j==len(res):
            
                r=r+1
    
    
    print(len(res))
    tsp2= time.time()
    print('time',tsp2-tsp1)
    fichier = open("res.txt", "w")
    for res1 in res:
        for k in res1:
            fichier.write(str(k))
            fichier.write('\t')
        fichier.write('\n')
    
    fichier.close()
    #res=[[1, 3, 9, 10, 12, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 20], [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 22, 33]]
    #res=[[3, 4, 5, 9, 11, 12, 16, 17, 19, 20, 21, 22, 24, 25, 29, 30, 31, 36, 43, 46, 48, 51, 52, 56, 60], [1, 13, 15, 16, 17, 21, 22, 34, 35, 37, 38, 39, 40, 41, 44, 45, 47, 50, 51, 53, 54, 59, 62], {2, 6, 7, 8, 10, 14, 18, 20, 23, 26, 27, 28, 31, 32, 33, 42, 49, 55, 57, 58, 61}]
    #res=[[19, 4, 5, 9, 12, 46, 16, 51, 52, 22, 25, 24, 36, 60, 30, 56], [13, 15, 17, 21, 34, 35, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 50, 51, 53, 54, 59, 62], [2, 6, 7, 10, 14, 18, 23, 26, 27, 28, 32, 33, 42, 49, 55, 57, 58, 61], [1, 43, 11, 48, 3, 29, 31], [8, 55, 20, 31]]
    O=[]
    re=[]
    for k in res:
        re.extend(k)
    #print(re)
    o2=[]
    for i in G.nodes():
        cpt=re.count(i)
        O.append(cpt)
        if cpt>1:
            o2.append(i)
    print(o2)
    
    print(len(O))
    N=len(G.edges())
    m=0
    #res=[[2, 26 ,34, 38, 46 ,90, 104 ,106, 110] ,[3, 7 ,14, 16, 33, 40, 48 ,61 ,65 ,101, 107] ,[4 ,6 ,11, 41 ,53 ,73, 75 ,82, 85, 99, 103, 108] ,[1 ,5 ,10, 17, 24 ,42, 94, 105] ,[12, 25, 29, 51, 70, 91 ],[8, 9 ,22 ,23 ,52, 69 ,78 ,79 ,109, 112 ],[13, 15, 19 ,27 ,32, 35, 37, 39, 43, 44, 55, 62 ,72 ,86, 100] ,[47, 50, 54, 59 ,68 ,74 ,84 ,89 ,111 ,115 ],[20, 30, 31, 36 ,56, 80, 81, 83, 95, 102 ],[18 ,21 ,28 ,57 ,60, 63 ,64, 66, 71, 77, 88, 96, 97, 114] ,[45, 49 ,58, 67 ,76, 87, 92, 93, 98, 113 ]]
    #[[2 ,3, 4 ,8 ,9 ,10 ,14 ,15, 16, 19, 21, 23, 24, 25, 26, 27 ,28 ,29, 30, 31 ,32, 33, 34],[1, 2, 3, 4 ,5, 6, 7, 8 ,11, 12, 13, 14 ,17, 18, 20, 22]] 

    #[[2 ,3, 4 ,8 ,9 ,10 ,14 ,15, 16, 19, 21, 23, 24, 25, 26, 27 ,28 ,29, 30, 31 ,32, 33, 34],[1, 2, 3, 4 ,5, 6, 7, 8 ,11, 12, 13, 14 ,17, 18, 20, 22]] 

    #m=0
    #rr=[]
    for i in res:
        print(i)
    #    #rr.extend(list(i))  
        m=m+modu1(G,list(i),N,O)
        
    #print(m)
    #m=m/(2*N)
    #m=0
    
    #m=modu1(G,N,res)
        
    #print(m)
    m=m/(2*N)
    print(m)
