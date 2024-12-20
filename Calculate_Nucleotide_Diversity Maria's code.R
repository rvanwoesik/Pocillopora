#Calculate nucleotide diversity

#packages
library(adegenet)
library(Biostrings)
library(ape)
library(msa)
library(pegas)

#Read in files
EG_bin<- fasta2DNAbin("data/Clean Ends.fasta", quiet=FALSE, chunkSize=10, snpOnly=FALSE)

rownames(EG_bin)[1:3]=paste("Zanzibar")
rownames(EG_bin)[4]=paste("Mombasa")
rownames(EG_bin)[5:6]=paste("Tanga")
rownames(EG_bin)[7:8]=paste("Mombasa")
rownames(EG_bin)[9]=paste("Kisite")
rownames(EG_bin)[10]=paste("Mombasa")
rownames(EG_bin)[11:13]=paste("Mafia")
rownames(EG_bin)[14:17]=paste("Pemba")
rownames(EG_bin)[18]=paste("Mtwara")
rownames(EG_bin)[19]=paste("Mauritius")
rownames(EG_bin)[20]=paste("Zanzibar")
rownames(EG_bin)[21]=paste("Kanamai")
rownames(EG_bin)[22:24]=paste("Zanzibar")
rownames(EG_bin)[25:27]=paste("Mauritius")
rownames(EG_bin)[28]=paste("Pemba")
rownames(EG_bin)[29:30]=paste("Mombasa")
rownames(EG_bin)[31]=paste("Zanzibar")
rownames(EG_bin)[32]=paste("Mafia")
rownames(EG_bin)[33]=paste("Malindi")
rownames(EG_bin)[34:35]=paste("Kanamai")
rownames(EG_bin)[36:43]=paste("Mauritius")
rownames(EG_bin)[44:49]=paste("Bise")
rownames(EG_bin)[50]=paste("MotobuPort")
rownames(EG_bin)[51:53]=paste("Zamai")
rownames(EG_bin)[54:56]=paste("Aka")
rownames(EG_bin)[57]=paste("Bise")
rownames(EG_bin)[58]=paste("Tokashiki")
rownames(EG_bin)[59]=paste("PenghIsalnd")
rownames(EG_bin)[60:90]=paste("Sanya")
rownames(EG_bin)[91:105]=paste("Luhuitou")
rownames(EG_bin)[106:120]=paste("Houhai")
rownames(EG_bin)[121:124]=paste("GBR")
rownames(EG_bin)[125]=paste("LongReef")
rownames(EG_bin)[126:127]=paste("OrpheusIsland")
rownames(EG_bin)[128:129]=paste("WallaceIslet")
rownames(EG_bin)[130]=paste("NightReef")
rownames(EG_bin)[131]=paste("OrpheusIsland")
rownames(EG_bin)[132]=paste("WallaceIslet")
rownames(EG_bin)[133]=paste("NightReef")
rownames(EG_bin)[134]=paste("OrpheusIsland")
rownames(EG_bin)[135]=paste("LongReef")
rownames(EG_bin)[136]=paste("WallaceIslet")
rownames(EG_bin)[137:141]=paste("GreatDetachedReef")
rownames(EG_bin)[142]=paste("LongReef")
rownames(EG_bin)[143]=paste("OrpheusIsland")
rownames(EG_bin)[144:145]=paste("TydemanReef")
rownames(EG_bin)[146:148]=paste("SolitaryIsland")
rownames(EG_bin)[149]=paste("LongReef")
rownames(EG_bin)[150]=paste("SolitaryIsland")
rownames(EG_bin)[151]=paste("LongReef")
rownames(EG_bin)[152]=paste("WallaceIslet")
rownames(EG_bin)[153:154]=paste("NightReef")
rownames(EG_bin)[155]=paste("WallaceIslet")
rownames(EG_bin)[156]=paste("OrpheusIsland")
rownames(EG_bin)[157:161]=paste("LongReef")
rownames(EG_bin)[162:168]=paste("LordHoweIsland")
rownames(EG_bin)[169]=paste("GreatDetachedReef")
rownames(EG_bin)[170:171]=paste("LordHoweIsland")
rownames(EG_bin)[172:173]=paste("WallaceIslet")
rownames(EG_bin)[174:177]=paste("RibReef")
rownames(EG_bin)[178]=paste("SolitaryIsland")
rownames(EG_bin)[179:185]=paste("WallaceIslet")
rownames(EG_bin)[186:201]=paste("Happiti")
rownames(EG_bin)[202:227]=paste("PapeeteM")
rownames(EG_bin)[228:250]=paste("Tiahura")
rownames(EG_bin)[251:260]=paste("Avera")
rownames(EG_bin)[261:269]=paste("TeAvaPiti")
rownames(EG_bin)[270:274]=paste("UtuFara")
rownames(EG_bin)[275:300]=paste("Faaha")
rownames(EG_bin)[301:308]=paste("Tapaumu")
rownames(EG_bin)[309:332]=paste("Vaitoare")
rownames(EG_bin)[333:342]=paste("PapeeteT")
rownames(EG_bin)[343]=paste("Vaitoare")
rownames(EG_bin)[344:355]=paste("PapeeteT")
rownames(EG_bin)[356:369]=paste("Tautira")
rownames(EG_bin)[370:398]=paste("Vairao")
rownames(EG_bin)[399:425]=paste("KaneoheBay")
rownames(EG_bin)[426]=paste("MidwayAtoll")
rownames(EG_bin)[427]=paste("PearlHermes")
rownames(EG_bin)[428]=paste("MidwayAtoll")

nuc_div = data.frame(Location = numeric(0), nuc.div = numeric(0))
for (n in unique(rownames(EG_bin))) {
  Sites <- EG_bin[grep(as.character(n), rownames(EG_bin)),]
  
  if (nrow(Sites) >= 3) {
    ndiv_Sites <- nuc.div(Sites)
    x = data.frame(Location = n, nuc.div = as.numeric(ndiv_Sites))
    nuc_div = rbind(nuc_div, x)
  }
}

write.csv(nucleotide_diversity, "data/nuc_div.csv")

#Check diversity for specific sites
KB<- EG_bin[grep(as.character("KaneoheBay"), rownames(EG_bin)),]
LHI<- EG_bin[grep(as.character("LordHoweIsland"), rownames(EG_bin)),]
RR<- EG_bin[grep(as.character("RibReef"), rownames(EG_bin)),]
