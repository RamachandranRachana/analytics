function[]=dtw_classify(training_file,test_file)
[class_tr,obj_tr,dcol1_tr,dcol2_tr,train_lenght]=clssify(training_file);
[class_test,obj_test,dcol1_test,dcol2_test,test_lenght]=clssify(test_file);
caccuracy=0;
disp(train_lenght);
for e=1:test_lenght
   % disp('---');
   a=dcol1_test(e,1:end);   
   a=transpose(a);
   b=dcol2_test(e,1:end);
   b=transpose(b);
   y=horzcat(a,b);
   y(all(y==0,2),:)=[];
   n=size(y,1);
   %disp(x);
   for f=1:train_lenght
       a_tr=dcol1_tr(f,1:end);
       a_tr=transpose(a_tr);
       b_tr=dcol2_tr(f,1:end);
       b_tr=transpose(b_tr);
       x=horzcat(a_tr,b_tr);
       x(all(x==0,2),:)=[];
       %disp(y);
       m=size(x,1);
       %disp(m);
       %disp(n);
       c=zeros(m,n);
      %disp(x);
      %disp(y);
       %disp(x(1,1));
       %disp(x(1,2));
       %disp(y(1,1));
       %disp(y(1,2));
       c(1,1)=cost(x(1,1),y(1,1),x(1,2),y(1,2));
       %disp(c(1,1));
       for i=2:m
         c(i,1)=c(i-1,1)+cost(x(i,1),y(1,1),x(i,2),y(1,2));
         %disp(c(d,1));
       end
      for j=2:n
          c(1,j)=c(1,j-1)+cost(x(1,1),y(j,1),x(1,2),y(j,2));
      end
      for p=2:m
          for q=2:n
              c(p,q)=min([c(p-1,q) c(p,q-1) c(p-1,q-1)])+cost(x(p,1),y(q,1),x(p,2),y(q,2));
          end
      end
      %disp(c);
      %disp(transpose(c));
      hm(f,1)=c(m,n);
      hm(f,2)=class_tr(f);
   end
   val=sortrows(hm,1);
   distance=val(1,1);
   predicted=val(1,2);
   true=class_test(e);
   %disp(predicted);
   if true==predicted
          accuracy=1;
   else
          accuracy=0;
   end
   caccuracy=caccuracy+accuracy;
   fprintf('ID=%5d, predicted=%3d,true=%3d, accuracy=%4.2f, distance = %.2f\n',e,predicted,true,accuracy,distance);
end
fprintf('classification accuracy=%6.4f\n', caccuracy/test_lenght);
end