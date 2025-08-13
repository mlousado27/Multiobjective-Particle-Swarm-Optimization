# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:47:02 2023

@author: marti
"""

# COM ELITISMO --- TOP_Nparticles


# GENERALIZADO:
    # VARIAVEIS DE ENTRADA ILIMITADAS AUTOMATICO --- ATRAVES DE LEN(BOUNDS), BASTA ADICIONAR AS BOUNDS DESSA VARIAVEL ADICIONAL À LISTA DE BOUNDS ---> CRIADA NO FIM
    # FUNCOES OBJETIVO ILIMITADAS --- ATRAVES DA LISTA OBJETIVOS ---> CRIADA NO FIM, SAO EM FUNCAO DO VETOR QUE ENGLOBA TODAS AS VARIAVEIS DE ENTRADA
    # RESTRICOES ILIMITADAS --- ATRAVES DA LISTA RESTRICOES NA FUNCAO DE CALCULO DO SCORE ---> CRIADA MANUALMENTE CONSOANTE O QUE FAÇA SENTIDO




import random
import numpy as np
import matplotlib.pyplot as plt
import math


# ONDE DIZ AQUIIIIIIIIIIII SÃO OS SITIOS NECESSÁRIOS A ALTERAR CASO SE QUEIRA TER MAIS FUNÇOES (F1, F2, F3, F4, ...), PARA ALEM DE ADICIONAR ESSAS TAIS FUNCOES NOS INITS
# O PLOT É QUE NÃO DÁ PARA FAZER AQUI ACHO EU. EM BAIXO ESTÁ APENAS PARA 2 FUNCOES
# BASTA FAZER CTRL+F DE "AQUIIIIII" PARA VER OS SITIOS DE MUDANÇA. SAO 16 LOCAIS


# A DIMENSAO DAS FUNCOES ESTA AUTOMATIZADA. O NUMERO DE FUNCOES PARA CONSTITUIR OS "PAIRS" É QUE NAO


class Particle:
    def __init__(self, bounds): #, dim): AQUIIIIIIIIIIIIIIIIIIIII
        
        #self.dim=len(bounds) # MELHOT QUE TER XMAX E XMIN
        # bounds=[[xmin,xmax],[ymin,ymax],...]  ESTÁ GENERALIZADO!!!!!!
        
        
        
        # ESTE SELF.POSITION NAO VAI ESTAR A SER ALTERADO
        
        self.position=[0]*len(bounds)
        self.velocity=[0]*len(bounds)
        for i in range(len(bounds)):
            self.position[i]=random.uniform(bounds[i][0], bounds[i][1])
            self.velocity[i]=random.uniform(-1, 1)
            

        self.best_position = self.position
        self.best_fitness = -float('inf') # NOTE-SE QUE COM O SCORE TEMOS UM PROBLEMA DE MAXIMIZAÇÃO
        self.fitness = None
        
        # NOVO
        
        # AQUIIIIIIII
        
        
        #    self.pairs=[[self.f1(self.position), self.f2(self.position)]]  # AQUIIIIIIIIIIIIIIIIIIIII. PARA UMA PARTICULA, A CADA (X,Y) QUE VÁ TENDO VAI CORRESPONDER UM VALOR DE F1 E DE F2 QUE FORMAM UM PAR. ESTES PARES TODOS ESTÃO A SER ARMAZENADOS AQUI
                                                                        
        # self.best_pair = None # MELHOR PAR PARA CADA PARTICULA. CUIDADO COM O NONE ..... SEM USO
        
        #self.history = [[self.position], [self.fitness], [self.velocity], []]   #[pos, fit/score, vel, iteracao dessa particula]
        
        #    self.history = [self.position]
        
        
    
    
    
    
    
def update_position(position, velocity, bounds):
    
    new_position = [0]*len(bounds)
    
    for i in range(len(position)):
        new_position[i] = max(bounds[i][0], min(bounds[i][1],position[i] + velocity[i]))
        
    return list(new_position)
    
    
    
    
    
    
    
def update_velocity(position, velocity, particle_base, global_best_position, w, c1, c2):
    
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    
    
    new_velocity=[0]*len(bounds)

    for i in range(len(velocity)):
        new_velocity[i] = w * velocity[i] + c1 * r1 * (particle_base.best_position[i] - position[i]) + c2 * r2 * (global_best_position[i] - position[i])
    
    
    return list(new_velocity)
    
    
    
    
    
def update_fitness(pair, function, particle_base, position, historico_total, n_iterations, n_particles):
    # self.fitness = function(self.f1(self.position), self.f2(self.position)) ESTAS DUAS LINHAS SAO PARA O MODULE
    # self.pairs.append([self.f1(self.position), self.f2(self.position)])
    
    
    all_pairs = []                  # NECESSARIO PARA A FUNÇAO SCORE

    for particle in historico_total:
        all_pairs.append(particle[3])
    
    # PAIR = [f1(position), f2(position)] = OBJECTIVES                                              # AQUIIIIIIIIIIIIIIIIIIII
    
    fitness = function(pair, all_pairs, n_iterations, n_particles)
    
    if fitness > particle_base.best_fitness:                # O SENTIDO MUDA POIS QUEREMOS MAXIMIZAR O SCORE
        particle_base.best_position = position
        particle_base.best_fitness = fitness
        
        #self.best_pair = [self.f1(self.position), self.f2(self.position)]      SEM UTILIDADE   # AQUIIIIIIIIIIIIIIIIIIII
        
        
    return fitness
            
            
            
            
            
            
class PSO:
    def __init__(self, function, bounds, w, c1, c2, n_particles, n_iterations, funcoes_objetivo): #AQUIIIIIIIIIIIIIIII
        self.function = function
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.global_best_position = []
        self.global_best_fitness = -float('inf')
        
        # NOVO
        
        self.global_best_pair = []
        
        # self.f1=f1
        # self.f2=f2
        # AQUIIIIIIIIII
        self.funcoes_objetivo = funcoes_objetivo
        
        
        self.all_pairs_global= []
        self.dict_pair_rank = dict()


        self.history_all = []
        
        # GERAÇÃO DE PARTICULAS INICIAIS
        
        #self.particles = [Particle(bounds, f1, f2) for i in range(n_particles)] # AQUIIIIIIIIIIIIIIII
        
        for i in range(n_particles):
            particle=Particle(bounds)
            
            # OBJECTIVES = [f1(vetor), f2(vetor), f3(vetor), ...]
            objectives = []
            for function in funcoes_objetivo:
                objectives.append(function(particle.position))
                
            
            self.history_all.append([particle, particle.position, particle.velocity, objectives, particle.fitness])
            
            # a particula inicial faz parte do historico pq assim conseguimos aceder a esta e atualizar os particles bests para a velocidade e assim e desta maneira nao faz mal se houverem varias particulas com a mm particula base
        
        
        
        
        
    def search(self):
        for i in range(self.n_iterations):
            
            for j in range(len(self.history_all)):
                
                self.history_all[j][4] = update_fitness(self.history_all[j][3], self.function, self.history_all[j][0], self.history_all[j][1], self.history_all, self.n_iterations, self.n_particles)
                
                if self.history_all[j][4] > self.global_best_fitness:
                    self.global_best_position = list(self.history_all[j][1])
                    self.global_best_fitness = self.history_all[j][4]
                    self.global_best_pair = list(self.history_all[j][3])
                    
            
                    
            # TIRAR PONTOS QUE POR ALGUMA RAZAO TÊM A MM POSIÇÃO
            
            historico_2 = []
            
            positions_all = []
            
            for particle in self.history_all:
                if particle[1] not in positions_all:
                    positions_all.append(particle[1])
                    historico_2.append(particle)
                
            self.history_all = historico_2
            

            
            # TOP_Nparticles
                
            all_fitnesses = []
            
            for particle in self.history_all:
                all_fitnesses.append(particle[4])
                
            fitness_aux = np.flip(sorted(all_fitnesses))
            
            historico_3 = []
            
            for fit in fitness_aux:
                for particle in self.history_all:
                    if particle[4] == fit:
                        historico_3.append(particle)
                        break
                
            self.history_all = historico_3 # todo e sorted pelo fitness
            
            # print('')
            # print(self.history_all)
            # print('')
            
            #################
            
                            
            # FAZER O PLOT A CADA ITERAÇÃO PARA VER A CURVA A APROXIMAR DO ZERO,0 ----- NAO PARECE ESTAR A CONVERGIR PARA 0
            
            # NO PLOT PROJETAMOS EM 2D OS 2 OBJETIVOS QUE QUEREMOS, BASTA ALTERAR XLINE E YLINE
            
            pareto = self.pareto_front_pair_rank()
            
            xline_all_values=[]
            yline_all_values=[]
            
            for rank,objectives in zip(pareto.values(),pareto.keys()):
                if rank == 1:
                    xline_all_values.append(objectives[0])
                    yline_all_values.append(objectives[1])
            
            
            print('------')
            print(xline_all_values)
            print('')
            print(yline_all_values) 
            print('------')
            
            plt.plot(xline_all_values, yline_all_values,'o')
    
            plt.xlim(0, 15)  
            plt.ylim(0, 15)  
            
            plt.show()
                   
            
            ################
            
            
            # UPDATE DO TOP30
            
            if i == 1:
                n = len(self.history_all)       # SE SE REMOVEREM ITENS POR ESTAREM DUPLICADOS NA PRIMEIRA ITERAÇÃO PODEMOS TER MENOS PARTICLES QUE AS 30
            else:
                n = self.n_particles
                
            for k in range(n):          # CRIAÇÃO DAS NOVAS PARTICULAS
                
                particle_base = self.history_all[k][0]
                velocity = update_velocity(self.history_all[k][1], self.history_all[k][2], self.history_all[k][0], self.global_best_position, self.w, self.c1, self.c2)
                position = update_position(self.history_all[k][1], velocity, self.bounds)
                
                #pair = [self.f1(position), self.f2(position)] = objectives           
                
                objectives = []
                for function in self.funcoes_objetivo:
                    objectives.append(function(position))
                
                fitness = None
                
                self.history_all.append([particle_base, position, velocity, objectives, fitness])
                

                
            # # FAZER O PLOT A CADA ITERAÇÃO PARA VER A CURVA A APROXIMAR DO ZERO,0 ----- NAO PARECE ESTAR A CONVERGIR PARA 0
            
            # # NO PLOT PROJETAMOS EM 2D OS 2 OBJETIVOS QUE QUEREMOS, BASTA ALTERAR XLINE E YLINE
            
            # pareto = self.pareto_front_pair_rank()
            
            # xline_all_values=[]
            # yline_all_values=[]
            
            # for rank,objectives in zip(pareto.values(),pareto.keys()):
            #     if rank == 1:
            #         xline_all_values.append(objectives[0])
            #         yline_all_values.append(objectives[1])
            
            
            
            # plt.plot(xline_all_values, yline_all_values,'o')
    
            # plt.xlim(0, 15)  
            # plt.ylim(0, 15)  
            
            # plt.show()
            
            


            
        return self.global_best_position, self.global_best_pair
    
    

        
    def pareto_front_pair_rank (self):
        
        self.dict_pair_rank ={}

        all_pairs=[]
        

        for particle in self.history_all:
            all_pairs.append(particle[3])
            
        all_pairs=sorted(all_pairs)         # ORDENA PELO PRIMEIRO MEMBRO DE CADA PAR. FICA MAIS ORGANIZADO
                                            # ALL_PAIRS NÃO ESTÁ A SER USADO PARA NADA, SIMPLESMENTE MANTEM A LISTA ORIGINAL DE TODOS OS PARES, QUE NO PAIRS_BK VAI SER CORTADA



        pairs_bk=[]                         # TIRAR DUPLICADOS
        for pair in all_pairs:
            if pair not in pairs_bk:
                pairs_bk.append(pair)
                
        self.all_pairs_global = pairs_bk    # GUARDO OS VALORES DOS PARES TODOS NO SELF PARA DEPOIS FAZER O GRAPH CASO QUEIRA
        
        # print(pairs_bk)
        # print(' ')
                
        n=1
        while n <= self.n_iterations * self.n_particles and len(pairs_bk) > 0:
            pareto=[]
            for pair in pairs_bk:
                dominant = True             # ASSUMO QUE É DOMINANTE E DPS SE ALGUM OUTRO PAR O DOMINAR, ENTAO DEIXA DE SER DOMINANTE
                for pair_2 in pairs_bk:
                    
                    if pair != pair_2 and pair_2[0] <= pair[0] and pair_2[1] <= pair[1]:     # O PAR(F1(POS),F2,F3,F4,...) É DOMINANTE SE NÃO FOR DOMINADO              AQUIIIIIIIII
                        dominant = False
                        break
                    
                    # is_dominated = True
                    # for i in range(len(pair)):
                    #     is_dominated = is_dominated and pair_2[i] < pair[i]
                        
                    # if is_dominated:
                    #     dominant = False
                    #     break
                    
                    
                if dominant:
                    pareto.append(pair)
                    
            for pair in pareto:
                self.dict_pair_rank[tuple(pair)] = n
            
        
            new_pairs=[]
            for pair in pairs_bk:
                if pair not in pareto:
                    new_pairs.append(pair)
            
            pairs_bk = new_pairs
            n += 1
                    
                
        # SORT THE DICTIONARY BY RANK. optional
        
        max_rank=max(self.dict_pair_rank.values())
        dict_sorted=dict()
        
        for i in range(max_rank):
            for rank,pair in zip(self.dict_pair_rank.values(),self.dict_pair_rank.keys()):
                if rank == i + 1:
                    dict_sorted[tuple(pair)] = i + 1
                    
        self.dict_pair_rank = dict_sorted
                    
                    
                    
        return self.dict_pair_rank       
    
    
    
 





# -------- FUNÇOES SCORE E AUXILIARES -----------------------------------------------------------------------------------



def score(point, list_points, n_iterations, n_particles):

#-----------------------------RANK-----------------------------------------

   
    # CRIAÇÃO DO DICIONÁRIO COM TODOS OS PONTOS E O SEU RESPETIVO RANK
    
    all_pairs=sorted(list_points)         # ORDENA PELO PRIMEIRO MEMBRO DE CADA PAR. FICA MAIS ORGANIZADO
                                        # ALL_PAIRS NÃO ESTÁ A SER USADO PARA NADA, SIMPLESMENTE MANTEM A LISTA ORIGINAL DE TODOS OS PARES, QUE NO PAIRS_BK VAI SER CORTADA
    # print(len(all_pairs))
    # print(all_pairs)
    
    # pairs_bk=[]                         # TIRAR DUPLICADOS
    # for pair in all_pairs:
    #     if pair not in pairs_bk:
    #         pairs_bk.append(pair)
    
    pairs_bk = all_pairs
            
    #self.all_pairs_global = pairs_bk    # GUARDO OS VALORES DOS PARES TODOS NO SELF PARA DEPOIS FAZER O GRAPH
           
    dict_pair_rank = dict()
    
    n=1
    while n <= n_iterations * n_particles and len(pairs_bk) > 0:
        pareto=[]
        for pair in pairs_bk:
            dominant = True
            for pair_2 in pairs_bk:
                
                # if pair_2[0] < pair[0] and pair_2[1] < pair[1]:     # AQUIIIIII 
                #     dominant = False
                #     break
                
                is_dominated = True
                for i in range(len(pair)):
                    is_dominated = is_dominated and pair_2[i] < pair[i]
                    
                if is_dominated:
                    dominant = False
                    break
                
                
            if dominant:
                pareto.append(pair)
                
        for pair in pareto:
            dict_pair_rank[tuple(pair)] = n
        
    
        new_pairs=[]
        for pair in pairs_bk:
            if pair not in pareto:
                new_pairs.append(pair)
        
        pairs_bk = new_pairs
        n += 1
                
                
    # CALCULAR O RANK DUM PONTO EM ESPECIFICO PESQUISANDO NO DICIONARIO QUE TEM TUDO
    
    rank_point=0            
    for rank,pair in zip(dict_pair_rank.values(),dict_pair_rank.keys()):
        if tuple(point) == pair:
            rank_point=rank
            

    #return dict_pair_rank, rank_point, max(dict_pair_rank.values()) - rank_point
    
    rank_score = max(dict_pair_rank.values()) - rank_point


#-------------------------CDA--------------------------------------------------------------

    #rank_1 = rank_point
    #dict_pairs = dict_pair_rank
    #dict_pairs = dict_rank(point, list_points, n_iterations, n_particles)   # PRECISO DO DICIONARIO PORQUE SO COMPARO AS DISTANCIAS NUM MESMO RANK
    
    #print(dict_pair_rank)
    
    # POINT = (f1(pos), f2(pos), f3(pos), ...)
    
    same_rank=[]
    
    for rank_2,pair_2 in zip(dict_pair_rank.values(),dict_pair_rank.keys()):
        if rank_2 == rank_point and pair_2 != tuple(point):
            same_rank.append(pair_2)
    
    dist_min=math.inf          # NESTE CONTEXTO FAZ SENTIDO FAZER ISTO?? CASO NO ÚLTIMO RANK SÓ HAJA UM PAIR ESTA DIST É 0??
    
    for pair in same_rank:
        
        #dist = (((pair[0]-point[0])**2)+((pair[1]-point[1])**2))**(1/2) # AQUIIIIIIIIIII
        
        total = 0
        for i in range(len(point)):
            total += (pair[i]-point[i])**2
            
        dist = total**(1/2)
            
        
        if dist < dist_min:
            dist_min = dist
            
            
            
    x = np.linalg.norm(point)
    
    if dist_min != math.inf:
        cda_1 = dist_min/(x + dist_min)
    else:
        cda_1 = 1       # ???????????????????????????????????????????? É ASSIM
    
    #print(cda_1)
    
    
#------------------RESTRICOES--------------------------------------


    count = 0
    
    for function in list1:
        if function(point) < 0:
            count += 1
    
    
#----------------FINAL-RETURN---------------------------------

    return rank_score + cda_1 - count


#------ FUNCOES PARA A RESTRICAO <---> APENDICE DA PARTE RESTRICOES DO SCORE ------------------

# def f1(point):
#     return point[0] + point[1]

# def f2(point):
#     return 3 * point[0] - point[1]**2

# def f3(point):
#     return point[0]**2 + point[1] - 1

def f4(point):
    return point[0]

#list1=[f1,f2,f3,f4]
list1=[f4]



# ---------------  FUNÇÔES TESTE OBJETIVO --------------------------------------------------------------------------------------------------------------------

# A ENTRADA DESTAS FUNCOES É UM VETOR, NESTE CASO É O POSITION, MAS SERÁ O "D" NO CASO DAS ENGRENAGENS

def sphere_function(position):
    return np.sum(np.power(position[0], 2))


def auckley(position):
    x=position[0]
    y=position[1]
    return -20 * np.exp(-0.2 * np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.exp(1)+20


def beale(position):
    x=position[0]
    y=position[1] 
    return ((1.5 - x + x*y)**2 + (2.25 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2)


def matyas(position):
    x=position[0]
    y=position[1] 
    return 0.26*(x**2 + y**2) - 0.48 * x * y


def rastrigin(position):
    A=10
    return A*2 + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in position])


def module(f1, f2):
    return np.sqrt(f1**2+f2**2)





# ----------  TESTE  -----------------------------------------------------------------------------------------------------------------------

# USAR AUCKLEY E MATYAS NAO SIGNIFICA MT, APENAS TENDE PARA 0 POIS AMBAS AS FUNCOES TÊM O MINIMO, 0, NAS COORDENADAS (0,0)
# BEALE nao tende para (0,0) por isso é mais interessante para testar com as outras, pois dá um range de solucoes, frente de pareto

objetivos = [auckley, beale]

bounds = [[-5, 5],[-4.5, 4.5]]

pso = PSO(function=score, bounds=bounds, n_particles=20, n_iterations=15, w=0.7, c1=1.5, c2=1.5, funcoes_objetivo=objetivos) # AQUIIIIII

best_position, best_pair = pso.search()

# pso.graph()


pareto = pso.pareto_front_pair_rank()

#Print the best position and fitness AND THE PAIRS + RANK

print('Best position = {}'.format(best_position))
print('Best pair = {}'.format(best_pair))

print('Pareto solutions = ' + str (pareto))







#ORDENAR O HISTÓRICO POR SCORE
#HISTORICO TEM QUE TER AS VELOCIDADES(2DIM TMB COMO A POS) GUARDADAS, O SEU SCORE, UM INDICE DE PARTICULA (CRIAR UM ID INICIAL POR PARTICULA, PARA AS TRAQUEAR)
#O SCORE DE TODOS MUDA EM TODAS AS ITERAÇÕES E DPS É QUE SAI O TOP 30

#PEGAR NO TOP_N_PARTICLE(30 pe) e executar as funçoes do PSO

#SE ELA ANTES DO UPDATE TIVER UM SCORE MELHOR MUDA À MESMA OU FICA COM OS SEUS REGISTOS DE MELHOR SCORE?!

# QUESTAO??????!!!!!!!!!

##########################################################################################

# ESTAR NO TOP30 DE SCORE NAO É NECESSARIAMENTE RANK1, POR ISSO NAO APARECE NO DESENHO?

#VER: ONDE SE FAZ A FILTRAÇÃO DO TOP 30. A MEIO DO DEF SEARCH, PORQUE O SCORE TEM DE SER CALCULADO PARA TODOS À MM.
# A CONFUSAO ESTÁ EM ENTENDER. QUANDO UMA PARTICULA TEM DUAS OU MAIS VERSOES SUAS QUE ESTAO DENTRO DO TOP30, ESSAS VERSOES NAO VAO SER ATUALIZADAS, CERTO? APENAS AQUELA QUE É A MAIS RECENTE. NAO COMPREENDO COMO É QUE, NA ITERAÇAO 20, POSSO IR ATUALIZAR, ISTO É APLICAR AS FUNCOES DE NOVA POSIÇAÕ E VELOCIDADE, À PARTICULA NAS SUAS VERSOES POR EXEMPLO DA ITERAÇÃO 12 E 17
# PQ A ATUALIZAÇÃO É FEITA À PARTICULA, À SUA VERSÃO MAIS RECENTE E NUMA ITERAÇÃO NAO FAZ MT SENTIDO ATUALIZAR DUAS VEZES OU MAIS A MM PARTICULA
# A ALTERNATIVA SERA ATUALIZAR AS PARTICULAS QUE TÊM A SUA ULTIMA VERSAO NO TOP 30??, MAS ASSIM SO VAO OCCORER ATUALIZAÇÕES ÀS VEZES PELO QUE O NUMERO TOTAL DE PARTICULAS NO FINAL VAI SER INFERIOR AO N_PRT * N_IT
# ALEM DISSO A VELOCIDADE DEVE SER GUARDADA APENAS O SEU ULTIMO VALOR PARA UMA PARTICULA, E ESTE VAI SENDO ATUALIZADO. OU GUARDA-SE POR APPEND E QND NAO HA ATUALIZAÇÃO DÁ-SE APPEND DUM VALOR REPETIDO. DESTA FORMA O VETOR DE VELOCIDADES FICA DO TAMANHO DO NUMERO DE ITERAÇOES?






