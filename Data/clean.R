# 为每个病人的动态数据生成自己的序列格式数据，保存在一个文件里面
# 所有时间窗都在里面
# 暂时要区分训练集和测试集，但也可以不区分


#  训练集


setwd("/home/luojiawei/multimodal_model/data/新的数据清洗 20211204/")
source("./工具函数1.R")
# 数据清理


# library(jsonlite)
library(data.table)
# library(readxl)
library(stringr)
# library(mltools)
library(magrittr)
library(lubridate)

print("开始读取数据")
print("ds1")
load("./raw_data/train/ds.1.tr.RData")

ds.1.tr[ds.1.tr[["病案号"]]=="IP0000908065","出院证明书:出院日期"]<-NA

print("ds3")
load("./raw_data/train/ds.3.tr.RData")
print("ds4")
load("./raw_data/train/ds.4.tr.RData")
print("ds5")
load("./raw_data/train/ds.5.tr.RData")

# PID<-ds.1.tr[["病案号"]][grep("(.+?)?急性(.+?)?胰腺炎(.+?)?",ds.1.tr[["出院主诊断名称"]])]
# PID<-ds.1[["病案号"]][grep("K85",ds.1[["出院主诊断代码"]]) & ds.1$年龄 >= 18]
load("./训练集 测试集 PID.RData")
length(PID)

PID_tr<-PID[tr_ind]
PID_te<-PID[te_ind]

print("开始读取字典")
load("./所有 字典.RData")
lab_vars<-lab_vars[!is.na(lab_vars[["是否选择"]]),]
drug_vars<-drug_vars[!is.na(drug_vars[["是否选择"]]),]
vit_vars<-vit_vars[!is.na(vit_vars[["是否选择"]]),]

# =======================================================

print("开始生成数据")
root<-"./clean_data/train" # 改成 test 就是测试集

for(k in 1:length(PID_tr)){
  # k<-5
  cat("正在处理",k," / ",length(PID_tr),"\n")
  
  x1.1<-ds.1.tr[ds.1.tr[["病案号"]]==PID_tr[k],]
  
  start_time<-x1.1[["一般情况 主诉 现病史 既往史:入院时间"]] %>% 
    gsub("[年月]","-",.) %>% gsub("日"," ",.) %>% ymd_hms(.)
  
  end_time<-x1.1[["出院证明书:出院日期"]] %>% 
    gsub("[年月]","-",.) %>% gsub("日"," ",.) %>% ymd_hms(.)
  
  if(is.na(end_time)){
    end_time<-x1.1[["出日期"]] %>%
      gsub("[年月]","-",.) %>% gsub("日"," ",.) %>% ymd(.) %>% as_datetime(.)
  }
  
  if(any(c(is.na(start_time), is.na(end_time)))) next
  if((end_time - start_time) < 24) next
  
  
  minute(start_time)<-0
  minute(end_time)<-0
  
  times<-start_time+hours(1:((end_time-start_time)*24))
  
  
  folder<-paste0(root,"/",PID_tr[k])
  if(!dir.exists(folder)){
    dir.create(folder)
    cat(folder,"不存在,已创建\n")
  } else{
    cat(folder,"已创建,跳过\n")
    next
  }
  
  x3.1<-ds.3.tr[ds.3.tr[["PAPMINO就诊号"]]==PID_tr[k],]
  x4.1<-ds.4.tr[ds.4.tr[["PADMNO就诊号"]]==PID_tr[k],]
  x5.1<-ds.5.tr[ds.5.tr[["体温单"]]==PID_tr[k],]
  if(nrow(x3.1)==0) next
  if(nrow(x4.1)==0) next
  if(nrow(x5.1)==0) next
  
  x3.1[["datetime"]]<-paste0(x3.1[["医嘱日期"]]," ",x3.1[["医嘱时间"]]) %>% ymd_hms(.)
  x3.1<-x3.1[!is.na(x3.1[["datetime"]]),]
  
  x4.1[["datetime"]]<-paste0(x4.1[["医嘱日期"]]," ",x4.1[["医嘱时间"]]) %>% ymd_hms(.)
  
  x5.1<-x5.1[order(x5.1[["ADMDR"]],x5.1[["日期"]],decreasing = F),]
  x5.1[["hour"]]<-rep(0,nrow(x5.1))
  
  
  
  indexs<-c()
  for(i in 2:nrow(x5.1)){
    if(x5.1[["ADMDR"]][i]==x5.1[["ADMDR"]][i-1]){ # 如果项目号一样
      if(x5.1[["日期"]][i]==x5.1[["日期"]][i-1]){ # 如果日期一一昂
        if(length(indexs)==0){ # 如果 index 里面没有元素
          indexs<-c(i-1,i) # 把前一行和当前行的索引加入
        } else{
          indexs<-c(indexs,i) # 如果已经有元素，那就只添加当前行的索引
        }
      } else{
        if(length(indexs) >= 2){ # 如果日期不一样，并且 index 里面有元素
		# 那就更新 hour 
          x5.1[["hour"]][indexs]<-round(seq(0,23,length.out = length(indexs)),0)
          indexs<-c() # 并且将 index  重新初始化
        }
      }
    } else{ # 如果项目号不一样，初始化 index
      indexs<-c()
    }
  }
  
  x5.1[["datetime"]]<-ymd(x5.1[["日期"]])+hours(x5.1[["hour"]])
  
  x3.total<-rbind(NULL) # time, 1+116
  x4.total<-rbind(NULL)
  x5.total<-rbind(NULL)
  label.total<-list()
  for(i3 in 1:7){
    label.total[[i3]]<-matrix(NA,nrow=length(times),ncol=length(lab_vars[["变量名"]])+3)
    # label.total[[i3]]<-rbind(NULL)
  }
  
  time_ind<-c()
  for(i1 in 1:length(times)){
    # i1<-2
    if(i1 %% 100 == 0){
      cat(i1,"/", length(times),"is done...\n")
    }
    cur_time<-times[i1]
    cur_time_range<-c(cur_time-minutes(60),cur_time)
	# rbind 保证是 n*m
    x3.2<-rbind(x3.1[x3.1[["datetime"]]>=cur_time_range[1] & x3.1[["datetime"]]<=cur_time_range[2],])
    x4.2<-rbind(x4.1[x4.1[["datetime"]]>=cur_time_range[1] & x4.1[["datetime"]]<=cur_time_range[2],])
    x5.2<-rbind(x5.1[x5.1[["datetime"]]>=cur_time_range[1] & x5.1[["datetime"]]<=cur_time_range[2],])
    if( all((nrow(x3.2)==0), (nrow(x4.2)==0), (nrow(x5.2)==0)) ) next else{
      # 如果三个都为空，直接跳过
	  time_ind<-c(time_ind, i1)
      
      x3.2.1<-rep(NA,nrow(lab_vars))
      if(nrow(x3.2)>0){
        for(i2 in 1:nrow(x3.2)){
          index<-which(x3.2[["化验结果项目编号"]][i2]==lab_vars[["化验结果项目编号"]])
          x3.2.1[index]<-as.numeric(str_extract(x3.2[["定量结果"]][i2], "[\\d.]+"))
        }
      }
      x3.total<-rbind(x3.total,x3.2.1)
      
      x4.2.1<-rep(0,nrow(drug_vars))
      if(nrow(x4.2)>0){
        for(i2 in 1:nrow(x4.2)){
          index<-which(x4.2[["医嘱项名称"]][i2]==drug_vars[["医嘱项名称"]])
          x4.2.1[index]<-1
        }
      }
      x4.total<-rbind(x4.total,x4.2.1)
      
      x5.2.1<-rep(NA, nrow(vit_vars))
      if(nrow(x5.2)>0){
        for(i2 in 1:nrow(x5.2)){
          index<-which(x5.2[["ADMDR"]][i2] == vit_vars[["ADMDR"]])
          x5.2.1[index]<-as.numeric(x5.2[["项目值"]][i2])
        }
      }
      x5.total<-rbind(x5.total, x5.2.1)
    }
    
    for(tw in 1:7){
      # cat("time window ",tw,"\n")
      pred_time_range<-c(cur_time,cur_time+days(tw))
      pred_time_range1<-c(cur_time-days(7),cur_time+days(tw))
      x3.3<-x3.1[x3.1[["datetime"]]>pred_time_range[1] & x3.1[["datetime"]]<=pred_time_range[2],]
      # x3.3[1:2,]
      x3.3.1<-rep(NA, nrow(lab_vars))
      if(nrow(x3.3)>0){
        for(i2 in 1:nrow(x3.3)){
          index<-which(x3.3[["化验结果项目编号"]][i2]==lab_vars[["化验结果项目编号"]])
          if(length(index)>0){
            if(is.na(x3.3.1[index])){
              x3.3.1[index]<-x3.3[["标示(上下箭头，低、正常、高)"]][i2]
            } else{
              if(x3.3.1[index]=="z"){
                x3.3.1[index]<-x3.3[["标示(上下箭头，低、正常、高)"]][i2]  
              } else next
            }
          }
        }
        x3.3.1<-ifelse(x3.3.1=="z",0,1)
      }
      
      x4.3<-x4.1[x4.1[["datetime"]]>pred_time_range[1] & x4.1[["datetime"]]<=pred_time_range[2],]
      diag<-any(x4.3[["医嘱项名称"]] %in% c("呼吸机辅助呼吸(有创)(间断)","呼吸机辅助呼吸(无创)","呼吸机辅助呼吸(有创)(持续)","呼吸机辅助呼吸（全）"))
      
      if(is.na(diag))
      {
        x3.3.1<-c(x3.3.1,NA)
      } else{
        if(diag) x3.3.1<-c(x3.3.1,1) else x3.3.1<-c(x3.3.1,0)
      }
      
      x5.3<-x5.1[x5.1[["datetime"]]>pred_time_range[1] & x5.1[["datetime"]]<=pred_time_range[2],]
      x3.4<-x3.1[x3.1[["datetime"]]>pred_time_range[1] & x3.1[["datetime"]]<=pred_time_range[2] & x3.1[["化验结果项目编号"]] == 3058,]
      # diag<-c(any(x5.3[["项目值"]][x5.3[["ADMDR"]] == "收缩压"] < 90),
      #         any(x5.3[["项目值"]][x5.3[["ADMDR"]] == "舒张压"] < 60),
      #         any(x4.3[["医嘱项名称"]] %in% c("FSM-呋塞米注射液","ZSYWSTD-注射用乌司他丁","TLSM-托拉塞米注射液","JQA-重酒石酸间羟胺注射液","XSYSLZZSY-硝酸异山梨酯注射液","YBSSXS-盐酸异丙肾上腺素注射液","QYXMHD-去乙酰毛花苷注射液")))
      diag<-c(any(x5.3[["项目值"]][x5.3[["ADMDR"]] == "收缩压"] < 90),
              any(x5.3[["项目值"]][x5.3[["ADMDR"]] == "舒张压"] < 60))
      diag1<-any(x4.3[["医嘱项名称"]] %in% c("FSM-呋塞米注射液","ZSYWSTD-注射用乌司他丁","TLSM-托拉塞米注射液","JQA-重酒石酸间羟胺注射液","XSYSLZZSY-硝酸异山梨酯注射液","YBSSXS-盐酸异丙肾上腺素注射液","QYXMHD-去乙酰毛花苷注射液"))
      diag2<-any(x3.4[["定量结果"]]>=2)
      diag<-diag & diag1 & diag2
      diag<-any(diag)
      if(is.na(diag))
      {
        x3.3.1<-c(x3.3.1,NA)
      } else{
        if(diag) x3.3.1<-c(x3.3.1,1) else x3.3.1<-c(x3.3.1,0)
      }
      
      x3.4<-x3.1[x3.1[["datetime"]]>pred_time_range1[1] & x3.1[["datetime"]]<=pred_time_range1[2] & x3.1[["化验结果项目编号"]] == 2930,]
      u1<-c();u1.1<-c();u2<-c();u3<-as.numeric(x3.4[["定量结果"]]);u4<-x3.4$datetime
      for (i3 in 1:(length(u3)-1)) {
        u1<-c(u1,-(u3[i3]-u3[(i3+1):length(u3)])/(u3[i3]+1e-8))
        u1.1<-c(u1.1,-(u3[i3]-u3[(i3+1):length(u3)]))
        u2<-c(u2,-(u4[i3]-u4[(i3+1):length(u4)]))
      }
      
      diag<-any(any(u1.1>=0.3*88.4 & u2<=48),any(u1>=0.5 & u2<=7*24))
      
      if(is.na(diag))
      {
        x3.3.1<-c(x3.3.1,NA)
      } else{
        if(diag) x3.3.1<-c(x3.3.1,1) else x3.3.1<-c(x3.3.1,0)
      }
      
      label.total[[tw]][i1,]<-x3.3.1
    }
  }
  
  # apply(label.total[[1]], 1, sum, na.rm=T)
  
  label.total.m<-list()
  for(i3 in 1:7){
    colnames(label.total[[i3]])<-c(lab_vars[["变量名"]],"res_f","cir_f","ren_f")
    rownames(label.total[[i3]])<-NULL
    label.total[[i3]]<-rbind(label.total[[i3]][time_ind,])
    label.total.m[[i3]]<-apply(label.total[[i3]],2,function(x)ifelse(is.na(x),0,1))
  }
  
  # dim(x3.total);dim(label.total[[3]])
  
  
  comb<-as.data.frame(cbind(rbind(x3.total),rbind(x5.total)))
  comb<-cbind("datetime"=times[time_ind],comb)
  
  names(comb)[2:ncol(comb)]<-c(lab_vars[["变量名"]],vit_vars[["变量名"]])
  rownames(comb)<-NULL
  
  colnames(x4.total)<-drug_vars[["变量名"]]
  xd<-as.data.frame(cbind("datetime"=times[time_ind],x4.total))
  
  # 数据填补
  final_data<-as.data.table(comb)
  final_data[["datetime"]]<-as.numeric(final_data[["datetime"]]-start_time)
  
  indexs<-grep("Lab_|Vit_",names(final_data))
  x.m<-rbind(final_data[,indexs,with=F] %>% apply(.,2,function(x)ifelse(is.na(x),0,1)))
  
  indexs<-grep("Lab_",names(final_data))
  for(i in 1:length(indexs)){
    final_data[[indexs[i]]]<-fill_NA(final_data[[indexs[i]]], final_data[["datetime"]])
  }
  
  indexs<-indexs[final_data[,indexs,with=F] %>% apply(.,2,function(x)all(is.na(x))) %>% which]
  if(length(indexs)>0){
    for(i in 1:length(indexs)){
      final_data[[indexs[i]]]<-lab_vars[["缺失值替代"]][which(names(final_data)[indexs[i]] == lab_vars[["变量名"]])]
    }
  }
  
  indexs<-grep("Vit_",names(final_data))
  for(i in 1:length(indexs)){
    final_data[[indexs[i]]]<-fill_NA(final_data[[indexs[i]]], final_data[["datetime"]])
  }
  
  indexs<-indexs[final_data[,indexs,with=F] %>% apply(.,2,function(x)all(is.na(x))) %>% which]
  if(length(indexs)>0){
    for(i in 1:length(indexs)){
      final_data[[indexs[i]]]<-vit_vars[["缺失值替代"]][which(names(final_data)[indexs[i]] == vit_vars[["变量名"]])]
    }
  }
  
  indexs<-grep("Lab_|Vit_",names(final_data))
  file_path<-paste0(folder,"/","x.csv")
  write.csv(final_data[,c(1,indexs),with=F],
            file_path, row.names=F)
  file_path<-paste0(folder,"/","m0.csv")
  write.csv(x.m,
            file_path,
            row.names = F)
  
  file_path<-paste0(folder,"/","xd.csv")
  write.csv(xd,
            file_path, row.names=F)
  
  file_path<-paste0(folder,"/","time.csv")
  write.csv(data.frame("time"=final_data[["datetime"]],
                       "delta"=c(0,diff(final_data[["datetime"]]))),
            file_path,
            row.names = F)
  
  for(i3 in 1:7){
    file_path<-paste0(folder,"/","y-tw",i3,".csv")
    write.csv(label.total[[i3]],
              file_path,
              row.names = F)
    file_path<-paste0(folder,"/","m1-tw",i3,".csv")
    write.csv(label.total.m[[i3]],
              file_path,
              row.names = F)
  }
  
}


