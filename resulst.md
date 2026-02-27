# alberi

metriche:

* qualitativa: pca
* quantitative: dendrogramma, loss

## in-order

* riorganizzazione emerge a 500 e migliora da mille (cluster più compatti)
* emergono triangoli a 1500 tra parent-chilren
* pc1 encoda la depth; pc2 encodes left/right subtrees

* dendrogramma:
  * depth 2 (4 nodi): bene da 150
  * depth 3 (8 nodi): bene da 150

nota: plot dendrogramma unico depth 1,2,3: ha imparato una nozione di depth (poiché aggrega coerentemente nodi della stessa depth) ma non una nozione globale di albero (altrimenti avrebbe clusterizzato seguendo la gerarchia). bene a 2000.

## shuffled

* riorganizzazione emerge a 500 e migliora da mille (cluster più compatti)
* emergono triangoli a 1500 tra prent-chilren
* pc1 encoda la depth; pc2 encodes left/right subtrees

* dendrogramma:
  * depth 2 (4 nodi): bene 150, male da 500
  * depth 3 (8 nodi): bene da 150

nota: plot dendrogramma unico depth 1,2,3. bene da 500.

* il comportamento simile di shuffled e in-order è confermato dalle loss simili

## only leaves

non abbiamo nè la loss, nè ha senso intepretare le pc

### in-order depth first

* dal dendrogramma si vede che riesce a clusterizzare subito correttamente 

### shuffled

* dal dendrogramma: funziona da 1500

### ordered

* non funziona mai

# cerchi

metriche:

* qualitativa: pca

* quantitativ1: loss (non informativa, da debuggare)

* NOTA: pc1 introduce pair sampling bias

## ordered

* pc2-pc3 trovano il cerchio approssimativamente bene già da 400, primo cerchio vero a 2000

## shuffled

* non trovo mai il cerchio 

# GRID

metriche:

* qualitativa: pca

* quantitativa: loss

## ordered

* pc1-pc2: fase transiente fino a l<800, griglia a 800 poi riorganizzazione

* loss: alta nella fase transiente, poi sempre minore
* interpretazione: la riorganizzazione pur non producendo una griglia pulita è tale da to solve effectively the ntp task

## shuffled

* pc1-pc2: non trovo mai la griglia
* loss: alta e maggiore di quella ordered. all'aumentare del contesto però va a zero dunque sole effectively ntp task pur senza riorganizzazione

# giorni

## albero (shuffled)

* risultati ok già a 150 (range 50-500)
* dendrogramma completo, trova l'albero globale a 150 (le rappresentazioni a 150 sembrano effettivamente simmetriche e coerenti); a range alti (500/1000/1500/2000) solita questione che recupera le gerarchie solo a parità di depth.
* pc1 encoda la depth in generale

## cerchio (orderd)

metriche:

* qualitativa: pca

* quantitativ1: loss (non informativa, sarebbe da debuggare)

* NOTA: pc1 introduce pair sampling bias

### ordered

* pc2-pc3 trovano il cerchio approssimativamente bene già da 300 (obs: ho meno punti)

### shuffled

* non trovo mai il cerchio 


# conclusioni

* tree are learned effectively in all the cases easely as confirmed by pca, dendrogram and losses. The global structure of the graph is not recovered but only the hiararchies (i.e. belonging to a certain depth), probably because that information is enough to solve the ntp task that we propted. We hypotize that this is a natural consequence of the icl mechanism, it only wants to put close things that are similar (neighbors). 

* the arrangement bias only holds for ordered/shuffled in circle and grid graphs, not for trees
* sampling strategy has a direct impact, as we observed for circles, the only dataset for wich we changed the prompt structure and the relashionship pairs in the input.
