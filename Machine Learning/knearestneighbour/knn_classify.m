function[]=knn_classify(training_file,test_file,k)
train_data=load(training_file);
test_data=load(test_file);
Ttrain=double(train_data(:,end));
Ttest=double(test_data(:,end));
train_data=double(train_data(:,1:end-1));
test_data=double(test_data(:,1:end-1));

train_mean=mean(train_data);
train_std=std(train_data,1);

train_normalize_data=normalize(train_data,train_mean,train_std);
test_normalize_data=normalize(test_data,train_mean,train_std);
%disp(train_normalize_data);
%disp('-----------------------------');
%disp(test_normalize_data);
caccuracy=0;
for i=1:size(test_normalize_data,1)
   d=test_normalize_data(i,:)-train_normalize_data;
   %disp(d);
   d=d.^2;
   %disp(d);
   d=sum(d,2);
   d=sqrt(d);
   %disp(d);
   hash_map=[d Ttrain];
   val=sortrows(hash_map,1);
   if k==1
       true=Ttest(i);
       predicted=val(1,2);
       if true==predicted
          accuracy=1;
       else
           accuracy=0;
       end
       caccuracy=caccuracy+accuracy;
       fprintf('ID=%5d, predicted=%3d,true=%3d, accuracy=%4.2f\n',i-1,predicted,true,accuracy);
   else
       true=Ttest(i);
       for m=1:k
          vec(m)=val(m,2); 
       end
       if (length(unique(vec)))==k||(length(unique(vec)))==1
           predicted=val(1,2);
       else
           predicted=mode(vec);
       end
       if true==predicted
          accuracy=1;
       else
           accuracy=0;
       end
       caccuracy=caccuracy+accuracy;
       fprintf('ID=%5d, predicted=%3d,true=%3d, accuracy=%4.2f\n',i-1,predicted,true,accuracy);
   end
   
end
fprintf('classification accuracy=%6.4f\n', caccuracy/size(test_normalize_data,1));

end