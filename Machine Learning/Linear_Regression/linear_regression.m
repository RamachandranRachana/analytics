function[]=linear_regression(filename,degree,lamda)
A=load(filename);
X= A(:,1);
T=A(:,2);
First=power(X,0);
Second=power(X,1);
Third=power(X,2);

if degree==1
    NewX=horzcat(First,Second);
    phi=vec2mat(NewX,2);
    phi=double(phi);
    
else
    NewX=horzcat(First,Second,Third);
    phi=vec2mat(NewX,3);
    phi=double(phi);
    %disp(phi);
end
tar=double(vec2mat(T,1));
if lamda==0      
    transphi=transpose(phi);
    firstmul=transphi*phi;
    secmul=inv(vpa(firstmul))*transphi;
    lastmul=secmul*tar;
    
    w0=round(lastmul(1)*10000)/10000;
    w1=round(lastmul(2)*10000)/10000;
    if degree==1
        w2=0;
    else
        w2=round(lastmul(3)*10000)/10000;
    end
    fprintf('w0=%.4f\n', w0);
    fprintf('w1=%.4f\n', w1);
    fprintf('w2=%.4f\n',w2);
    
else
    regterm=lamda*eye(degree+1);
    transphi=transpose(phi);
    firstmul=transphi*phi;
    firstsum=regterm+firstmul;
    secmul=inv(vpa(firstsum))*transphi;
    lastmul=secmul*tar;
    w0=round(lastmul(1)*10000)/10000;
    w1=round(lastmul(2)*10000)/10000;
    if degree==1
        w2=0;
    else
        w2=round(lastmul(3)*10000)/10000;
    end
    fprintf('w0=%.4f\n', w0);
    fprintf('w1=%.4f\n', w1);
    fprintf('w2=%.4f\n',w2);
end
end

