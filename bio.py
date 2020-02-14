import random ,time
from multiprocessing.pool import ThreadPool
from threading import Thread
import numpy as np



def fitness1(genom,extra=None):
    sumation =0
    for i in range(len(genom)):
        if i %2 ==0:
            sumation+=genom[i]
        else:
            sumation-=genom[i]
    return sumation*genom[0]

def parent_selection(fitness):
    minimun = abs(min(fitness))
    tmp = [ fitness[i] + minimun for i in range(len(fitness))]

    tmp2 = [0]

    for i in range(len(tmp)):
        tmp2.append(tmp2[i]+tmp[i])


    rnd = random.uniform(0,max(tmp2[-1],0))

    for i in range(len(tmp2)-1):
        if rnd >= tmp2[i] and rnd <= tmp2[i+1]:
            return i
    return len(tmp2)-1
    
    
def crossover(par1,par2):
    crosspoint = random.randint(0,len(par1)-1)

    member1 = [] 
    member2 = []
    for i in range(len(par1)):
        if i < crosspoint:
            member1.append(par1[i])
            member2.append(par2[i])
        else:
            member1.append(par2[i])
            member2.append(par1[i])
    return (member1,member2)

def mutation(par1):
    pos = random.randint(0,len(par1)-1)
    newindividual = par1
    newindividual[pos] = random.uniform(0,1)

    return newindividual

def crossoverR(par1,par2):
    crosspoint = random.randint(0,len(par1)-1)
    child = [] 
    for i in range(len(par1)):
        if i < crosspoint:
            child.append(par1[i])
        else:
            child.append(par2[i])
    return child


    

def genetic(inp):
   
    Pm = 0.01
    start_time = time.time()
    #population initialization
    popolation = inp["pop"]
    gen_size = inp["size"]
    fitness = inp["fit"]
    fitness_f = inp["file"]
    pop= [[ random.uniform(0,1) for i in range(gen_size)  ] for j in range(popolation)  ]
   # print(len(pop),popolation,pop)
    rng1 = np.random.RandomState(40)
    best_list = []
    The_best =None
    
    for i in range(100):
        #giving 11 hours to finish
        if (time.time() -start_time) > 11*60*60:
            fitness_f.write("TIMEOUT!\n")
            break
        print("generation: " +str(i+1))
        inp["seed"] = rng1.randint(0, 1000, 1)[0]
        
        
        #fitness_list = [None]* popolation
        fitness_list = []
        
        
        
        pool = ThreadPool(processes=popolation)
        fitness_list = [pool.apply(fitness, (pop[j],inp)) for j in range(popolation) ]
        #multithread
        #threads = [None]* popolation

#        def mulithreadFitness(genom,fit,results,index):
#            results[index] = fit(genom,inp)

       # for i in range(len(pop)):
        #    individual = pop[i]
        #    sync_result = pool.apply_async(fitness, (individual,inp))
            #threads[i] = Thread(target=mulithreadFitness,args=(individual,fitness,fitness_list,i))
            #threads[i].start()
            #fitness_list.append(fitness(individual,inp))
        #re = sync_result.get(0xffffff)
        #for r in re:
        #    fitness_list.append(r)
            

        best_list.append(max(fitness_list))
        if len(best_list)>3:
            best_list.pop(0)
        if best_list[1:] == best_list[:-1] and len(best_list)==3:
            Pm+=0.05
            print("Mutation: " + str(Pm))
        else:
            mutation=0.01
        
        maximun = max(fitness_list)
        best_fit = fitness(The_best,inp)
        fitness_f.write("Best: " + str(maximun) +"\n")
        if best_fit < maximun:
            for j in range(len(fitness_list)):
                if fitness_list[j] == maximun:
                    The_best= pop[j]

        print("\n\nBest: " + str(max(fitness_list)))
        newpop= []
        #time.sleep(1)
       # parents = [ parent_selection(fitness_list) for i in range(len(pop))]
        #newpop = [pop[parents[i]] for i in range(len(pop))]
       # if random.uniform(0,1) < Pc:
        #    for i in range(0,len(newpop),2):
        #        newmembers = crossover(newpop[i],newpop[i+1])
        #        newpop[i]=newmembers[0]
        #        newpop[i+1]=newmembers[1]
                    
        for j in range(popolation):
            x = parent_selection(fitness_list)
            y = parent_selection(fitness_list)
            newpop.append(crossoverR(pop[x],pop[y]))

        
        
        for individual in range(len(newpop)):
            if random.uniform(0,1) < Pm:
                newpop[individual] = mutation(newpop[individual])


       # print(newpop)
        pop= newpop 
        print("Population size: " + str(len(pop)))

 

    return The_best

