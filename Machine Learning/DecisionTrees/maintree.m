function[]=maintree(training_file,test_file,option,pruning_thr)
example=load(training_file);
example_test=load(test_file);
T=double(example(:,end));
example=double(example(:,1:end));

%disp(T);
nmax=max(T);
%disp(nmax);
%disp(size(distribution,1));
attributes=zeros(size(example,2)-1,1);
attributes=attributes+1;
for i=1:size(attributes,1)
   attributes(i,1)=attributes(i,1)*i; 
end
if isequal(option,'forest3')
   option='randomized'; 
   tree1=[];
   threshold1=[];
   gain1=[] ;
   tree2=[];
   threshold2=[];
   gain2=[] ;
   tree3=[];
   threshold3=[];
   gain3=[] ;
   [tree1,threshold1,gain1]=dtl(example,pruning_thr,option,attributes,nmax,tree1,threshold1,gain1,1); 
   [tree2,threshold2,gain2]=dtl(example,pruning_thr,option,attributes,nmax,tree2,threshold2,gain2,1);
   [tree3,threshold3,gain3]=dtl(example,pruning_thr,option,attributes,nmax,tree3,threshold3,gain3,1);
   for i=1:size(tree1,2)
        feature=tree1(:,i)-1;        
        
        if feature~=-1               
                
            if threshold1(:,i)==-1 & gain1(:,i)==-1
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',0,i,-1,threshold1(:,i),gain1(:,i)); 
            else
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',0,i,tree1(:,i)-1,threshold1(:,i),gain1(:,i));
            end
        end
   end
   for i=1:size(tree2,2)
        feature=tree2(:,i)-1;        
        
        if feature~=-1               
                
            if threshold2(:,i)==-1 & gain2(:,i)==-1
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',1,i,-1,threshold2(:,i),gain2(:,i)); 
            else
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',1,i,tree2(:,i)-1,threshold2(:,i),gain2(:,i));
            end
        end
   end
   for i=1:size(tree3,2)
        feature=tree3(:,i)-1;        
        
        if feature~=-1               
                
            if threshold3(:,i)==-1 & gain3(:,i)==-1
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',2,i,-1,threshold3(:,i),gain3(:,i)); 
            else
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',2,i,tree3(:,i)-1,threshold3(:,i),gain3(:,i));
            end
        end
   end
   caccuracy=0;
    for hh=1:size(example_test,1)
        index=1;
        flag=1;
%disp(example_test(1,attr));
        while flag==1
            attr=tree1(index);
            thr=threshold1(index);
            ga=gain1(index);
            if thr==-1 & ga==-1
                val1=attr;
                flag=0;
            else
                if (example_test(hh,attr))>=thr
                    index=(2*index)+1;
                else
                    index=(2*index);
                end
            end
        end
        index=1;
        flag=1;
        while flag==1
            attr=tree2(index);
            thr=threshold2(index);
            ga=gain2(index);
            if thr==-1 & ga==-1
                val2=attr;
                flag=0;
            else
                if (example_test(hh,attr))>=thr
                    index=(2*index)+1;
                else
                    index=(2*index);
                end
            end
        end
        index=1;
        flag=1;
        while flag==1
            attr=tree3(index);
            thr=threshold3(index);
            ga=gain3(index);
            if thr==-1 & ga==-1
                val3=attr;
                flag=0;
            else
                if (example_test(hh,attr))>=thr
                    index=(2*index)+1;
                else
                    index=(2*index);
                end
            end
        end
        Tt=example_test(hh,end);
        val1=int16(val1);
        val2=int16(val2);
        val3=int16(val3);
        atr1data=zeros(1,nmax+1);
        atr1data(val1+1)=1;
        atr2data=zeros(1,nmax+1);
        atr2data(val2+1)=1;
        atr3data=zeros(1,nmax+1);
        atr3data(val3+1)=1;
        tot=atr1data+atr2data+atr3data;
        tot=tot/3;
        [M,I]=max(tot);
        if (I-1)==Tt
            accuracy=1;
        else
            accuracy=0;
        end
        caccuracy=caccuracy+accuracy;
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',hh,I-1,Tt,accuracy);
    end
fprintf('classification accuracy=%6.4f\n',caccuracy/size(example_test,1));
else    
    tree=[];
    threshold=[];
    gain=[];
    [tree,threshold,gain]=dtl(example,pruning_thr,option,attributes,nmax,tree,threshold,gain,1); 
    %disp(size(tree));
    %disp(size(threshold));
    %disp(size(gain));
    for i=1:size(tree,2)
        feature=tree(:,i)-1;        
        
        if feature~=-1               
                
            if threshold(:,i)==-1 & gain(:,i)==-1
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',0,i,-1,threshold(:,i),gain(:,i)); 
            else
                fprintf('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n',0,i,tree(:,i)-1,threshold(:,i),gain(:,i));
            end
        end
    end
    %disp(example_test(1,1:end-1));
    caccuracy=0;
    for hh=1:size(example_test,1)
        index=1;
        flag=1;
%disp(example_test(1,attr));
        while flag==1
            attr=tree(index);
            thr=threshold(index);
            ga=gain(index);
            if thr==-1 & ga==-1
                Tt=example_test(hh,end);
                if attr==Tt
                    accuracy=1;
                else
                    accuracy=0;
                end
                caccuracy=caccuracy+accuracy;
                fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',hh,attr,Tt,accuracy);
                flag=0;
            else
                if (example_test(hh,attr))>=thr
                    index=(2*index)+1;
                else
                    index=(2*index);
                end
            end
        end
    end
    fprintf('classification accuracy=%6.4f\n',caccuracy/size(example_test,1));
%disp(size(example_test(1,:)));
%disp(tree);
end
end