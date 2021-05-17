#1-模式模拟二进制文件数据读取
readBin_zlh=function(
      file1="d:/BCC-CSM1.1-m.historical_1961-2005.grd",#模式数据
      file2="d:/obs_model_sta_latlon.csv", #模式对应的站点经纬度信息            
      file3='d:/BCC-CSM1.1-m.historical_1961-2005.csv',# 将二进制文件转换为excel可阅读格式
      size=4,n=300000000)           #这里不要修改
{
 sta_sn=read.table(file2,header=T,sep='') #读取模式站点经纬度信息
 ncol=length(sta_sn$sn) #站点个数
 col_name=c('year','mon','day',paste('X',sta_sn$sn,sep=''))
 zz<- file(file1, "rb") #读取模式模拟数据
 temp1=readBin(zz,double(),size=size,n=n)
 temp2=matrix(temp1,ncol=ncol+3,byrow=F) #要按列读取数据
 print(dim(temp2));
 close(zz)
 rlst=as.data.frame(temp2)
 names(rlst)=col_name
 rlst=rlst
 write.csv(rlst,file=file3,row.names=F)
}


# 读取模式历史时期模拟数据，并写出文件到file3
bcc_his=readBin_zlh(
  file1="d:/BCC-CSM1.1-m.historical_1961-2005.grd",#模式数据
  file2="d:/obs_model_sta_latlon.csv", #模式对应的站点经纬度信息            
  file3='d:/3_BCC-CSM1.1-m.historical_1961-2005.csv',# 将二进制文件转换为excel可阅读格式
  size=4,n=300000000)  
# 读取模式未来模拟数据，并写出文件到file3
bcc_fut=readBin_zlh(
  file1="d:/BCC-CSM1.1-m.rcp45_2006-2100.grd",#模式数据
  file2="d:/obs_model_sta_latlon.csv", #模式对应的站点经纬度信息            
  file3='d:/4_BCC-CSM1.1-m.rcp45_2006-2100.csv',# 将二进制文件转换为excel可阅读格式
  size=4,n=300000000)      



#2-Netcdf文件读写函数
library('ncdf4')
nc=nc_open(infile) # infile 为包含路径的文件名称
print(nc)          # 显示netcdf文件概况,了解数据维数、变量等信息
obs.data0=ncvar_get(nc,varname) # varname为拟读取的变量名称
dim(obs.data0)    
nc_close(nc)
rm('nc')




#----------------------------------------------------------------------------
pre_data_read_zlh=function(infile="D:/R_spatial/data/precip_daily_756sta_1951-2012.nc",                             varname='precip',outfile=NULL,yearb=1961,yeare=2005,season='JJA',    
                           obs.latlon=sta_latlon_756,sta_dr=c('北京','武汉','广州'),    
                           na.n=0,max.check=400,min.check=0,    
                           max.super=1500,min.super=0)     
  
  # infile="D:/R_spatial/data/precip_daily_756sta_1951-2012.nc";varname='precip';outfile=NULL
  # yearb=1961;yeare=2005;season='JJA'
  # obs.latlon=sta_latlon_756;
  # sta_dr=c('北京','武汉','广州')
  # max.check=400;min.check=0
  # max.super=1500;min.super=0
  # na.n=0
{library('ncdf4')
  library('matlab')
  tic(); print(Sys.time())
  #1--读取数据
  # nc=open.ncdf(infile)
  # obs.data0=get.var.ncdf(nc,varname)
  # dim(obs.data0)
  # close.ncdf(nc)
  # rm('nc')
  #对程序作了更新 2016-11-17
  nc=nc_open(infile)
  obs.data0=ncvar_get(nc,varname)
  dim(obs.data0)
  nc_close(nc)
  rm('nc')
  
  obs.data1=matrix(obs.data0,nrow=31*12*62)
  obs.data2=as.data.frame(obs.data1)
  obs.sta.name=paste("X",obs.latlon$sn,sep='')
  names(obs.data2)=obs.sta.name
  
  #2---添加时间信息
  year=rep(1951:2012,each=372,times=1)
  mon=rep(1:12,each=31,times=62)
  day=rep(1:31,times=62*12)
  obs.time=cbind(year,mon,day)
  
  obs.data=cbind(obs.time,obs.data2)
  
  rm('obs.data0','obs.data1',"obs.data2",'obs.sta.name',"year","mon","day")
  
  obs.latlon_dr_0=obs.latlon
  obs.data_dr_1=season_data_read_zlh(dat=obs.data,yearb=yearb,yeare=yeare,season=season)
  rm('obs.data','obs.latlon','obs.time')
  
  #2-缺测情况,删除整行均为缺测值的天（实际效果是删除闰月、月多出来得31日）
  hl1=dim(obs.data_dr_1)
  print(hl1)
  n1=hl1[2]-3#原始数据站点数
  t1=hl1[1]
  #2-1 #按时间去除一行全为缺测值的润年或多出来的月底日
  narow=apply(obs.data_dr_1,1,function(x) length(which(is.na(x)==T)))
  obs.data_dr_2=obs.data_dr_1[!(narow==n1),]
  rm('obs.data_dr_1','hl1','n1','t1','narow')
  
  hl2=dim(obs.data_dr_2)
  print(hl2)
  n2=hl2[2]-3#原始数据站点数
  t2=hl2[1]
  
  #2-2 #按列看每一个站缺测情况，去除缺测较多的站点
  nacol=apply(obs.data_dr_2[,4:(n2+3)],2,function(x) length(which(is.na(x)==T)))
  obs.data_dr_3=obs.data_dr_2[,c(T,T,T,nacol<=na.n)]
  "obs.latlon_dr"=obs.latlon_dr_0[nacol<=na.n,]
  rm('hl2','n2','t2','obs.data_dr_2','nacol','obs.latlon_dr_0')
  
  hl3=dim(obs.data_dr_3)
  print(hl3)
  n3=hl3[2]-3#原始数据站点数
  t3=hl3[1]
  print(n3)
  #2 评估降水量级，输出那些超大值，决定如何处理这些异常值
  for (i in 1:n3)
  {x=obs.data_dr_3[,i+3];
  c=which((x>max.check)|(x<min.check));
  if (length(c)>=1)
    print(c(i,c,as.vector(x[c])))#依次为站点序号、该站点异常值位置及对应的取值
  }
  rm('hl3','c','t3','x','i')
  print(obs.latlon_dr[match(sta_dr,obs.latlon_dr$name),])
  #依据前面情况异常数据处理：1：小于零的值以及-99.9 缺测处理；2、异常大值处理
  obs.data_dr_4=obs.data_dr_3
  for (i in 1:n3)
  {obs.data_dr_4[,i+3][is.na(obs.data_dr_3[,i+3])]=0 #将缺测赋值为0，后期再重新调整
  obs.data_dr_4[,i+3][obs.data_dr_3[,i+3]>=max.super]=obs.data_dr_3[,i+3][obs.data_dr_3[,i+3]>=max.super]/10#对于特殊超大数据，除以10，还原为合理值
  obs.data_dr_4[,i+3][obs.data_dr_3[,i+3]<min.super]=0  #对于特殊超大数据，除以10，还原为合理值
  }
  "obs.data_dr"=obs.data_dr_4
  rm('n3','obs.data_dr_3','obs.data_dr_4')
  print(dim(obs.data_dr))
  if(is.null(outfile)==F)
  {
    write.table(obs.data_dr,outfile,row.names=F,sep=',')
  }
  toc()
  # rlst=list(obs.data_dr=obs.data_dr,obs.latlon_dr=obs.latlon_dr)
  rlst=obs.data_dr
}
'model_time_read_zlh'=function(timeb="1961-01-01",timee="2005-12-31",yeap=2)
{ #timeb="1961-07-15";timee="2005-12-30";yeap=3
  if (yeap==1){
    timeb=as.Date(timeb);timee=as.Date(timee);
    time0=seq(timeb, timee, "day")
    year=as.numeric(substr(time0,1,4));
    mon=as.numeric(substr(time0,6,7));
    day=as.numeric(substr(time0,9,10))
    time=cbind(year,mon,day)     
  }  
  if (yeap==2) {
    timeb=as.Date(timeb);timee=as.Date(timee);
    time0=seq(timeb, timee, "day")
    year=as.numeric(substr(time0,1,4));
    mon=as.numeric(substr(time0,6,7));
    day=as.numeric(substr(time0,9,10))
    time1=cbind(year,mon,day)   
    time2=time1
    #time=time2[-which((time2[,2]==2)&(time2[,3]==29)),]
    time=time2[-((time2[,2]==2)&(time2[,3]==29)),]
  }
  if (yeap==3)
  {
    #  timeb=as.Date(timeb);timee=as.Date(timee);
    yearb=as.numeric(substr(timeb,1,4));yeare=as.numeric(substr(timee,1,4))
    year=rep(yearb:yeare,each=360)
    mon=sprintf('%02d',rep(1:12,each=30,times=length(yearb:yeare)))
    day=sprintf('%02d',rep(1:30,each=1,times=length(yearb:yeare)*12))
    time0=paste(year,mon,day,sep='-')
    if((is.na(match(timeb,time0))!=T)&(is.na(match(timee,time0))!=T))
    {
      time1=time0[(which(time0==timeb)):(which(time0==timee))]
      year=as.numeric(substr(time1,1,4));mon=as.numeric(substr(time1,6,7));
      day=as.numeric(substr(time1,9,10))
      time=cbind(year,mon,day)  
    } else {
      print("Warnings: Please check whether the input date is correct!")  
    }
  }
  time=time 
}
season_data_read_zlh=function(dat=sta_data_553,yearb=1961,
                              yeare=2005, season='JJA',
                              adyear.index=F)
{ seasonall=c("MAM","JJA","SON","DJF")
ind=match(season,seasonall) 
monb0=c(3,6,9,12) 
monb=monb0[ind];mone=(monb+3-1)%%12;  
print(c(monb,mone,seasonall[monb/3]))
rlst=month_data_read_zlh(dat=dat,yearb=yearb,yeare=yeare,     
                         monb=monb,mone=mone,adyear.index=adyear.index)
rlst
}
month_data_read_zlh=function(dat=sta_data_553,yearb=1961,
                             yeare=2005,monb=6,mone=8,
                             adyear.index=F)
{ print(c('Read data from:',c(monb,mone)))
  if(any(names(dat)[1:2]!=c('year','mon')))
  { print(names(dat)[1:2])
    print('Please check the names of time')
    names(dat)[1:2]=c('year','mon')  
  }
  dat_dr=dat[(dat$year>=yearb)&(dat$year<=yeare),]
  if (monb>mone) {
    dat_dr=dat_dr[((dat_dr$mon<=mone)
                   &(dat_dr$year>=yearb+1))
                  |((dat_dr$mon>=monb) 
                    &(dat_dr$year<=yeare-1)),]#跨年保持整年，不读取不完整数据
  } else {
    dat_dr=dat_dr[(dat_dr$mon<=mone)&(dat_dr$mon>=monb),]
  }
  if (monb>mone)  #跨年数据修正年份
  {
    if (adyear.index==T)
    { dat_dr[dat_dr$mon>=monb,'year']=dat_dr[dat_dr$mon>=monb,'year']+1                #对于跨年12前部分年份修正+1
    dat_dr=dat_dr[(dat_dr$year>=yearb+1)&(dat_dr$year<=yeare),]          
    } else{
      dat_dr=dat_dr #重新读取数据符合年限要求的数据
    }
  }
  ########################################################################################
  #因为后面提取因子采用的是按行名称提取，所以这里统一将数据行名称重新更新为新序列的序号
  #######################################################################################
  row.names(dat_dr)=1:length(dat_dr[,1])
  print("The time length of out_data:")
  print(length(dat_dr[,1]))
  dat_dr
}


fi1="D:/obs_model_latlon_756_revision.txt";
fi2="D:/precip_daily_756sta_1951-2012.nc"
"sta_latlon_756"=read.table(fi1,header=T)
"sta_data_553"=pre_data_read_zlh(infile=fi2, 
                                 outfile=NULL,
                                 #outfile="D:/R_spatial/out/obs_sta_jja.csv",
                                 yearb=1961,yeare=2005,season='JJA',
                                 obs.latlon=sta_latlon_756,sta_dr=c('北京','武汉','广州'),
                                 na.n=0,max.check=400,min.check=0,
                                 max.super=1500,min.super=0)  
'sta_name'=(names(sta_data_553))[-c(1:3)];
'sta_latlon_553_dr'=sta_latlon_756[sta_name,];
write.table(sta_latlon_553_dr,file='d:/553.csv',row.names=F,quote=F,sep=',')
