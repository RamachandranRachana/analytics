function[]=neural_network(training_file,test_file,layers,units_per_layer,rounds)
A=load(training_file);

nnmax=max(A);
nmax=max(nnmax);
A=A/nmax;
T=double(A(:,end));
T=T+0.01;
%disp(T);
u=max(T);
%disp(u);
%disp(T);
A=double(A(:,1:end-1));
bias=1;
alpha=1;
noofhiddenlayers=layers-2;
noofunitsoutputlayer=u*nmax;
unitslayer=units_per_layer;
if noofhiddenlayers~=0
    %outputweights=zeros(noofunitsoutputlayer,unitslayer);
    %outputweights=outputweights+0.5;
    outputweights = -0.05 + (0.05- (-0.05)).*rand(noofunitsoutputlayer,unitslayer);
    biass=zeros(noofunitsoutputlayer,1);
    biass=biass+1;
    outputweights=horzcat(outputweights,biass);
    biasi=zeros(size(A,1),1);
    numbers = 4;
    %--------------> 3layers
    if noofhiddenlayers<2    
        %disp(size(A,2));
        %weightfirst=zeros(unitslayer,size(A,2));
        %weightfirst=weightfirst+0.5;
        weightfirst = -0.05 + (0.05- (-0.05)).*rand(unitslayer,size(A,2));
        biases=zeros(unitslayer,1);
        biases=biases+1;
        weightfirst=horzcat(weightfirst,biases);
        biasi=biasi+1;
        A=horzcat(A,biasi);
        mm=0;
        %---------->TRAINING
        while mm<rounds
        %while mm<20
            mm=mm+1;
            for i=1:size(A,1)
            %for i=1:numbers
                %disp(weightfirst);
                %disp(A(i,1:end));
                %disp(transpose(A(i,1:end)));
                out1=weightfirst*transpose(A(i,1:end));  
                %out1=out1+bias;
                %disp(out1);
                for j=1:size(out1,1)
                    out1(j)=1/(1+exp(-out1(j))); 
                end
                %disp(out1);
                out1=vertcat(out1,1);
                %disp(out1);
                %disp(size(outputweights));
                %disp(size(out1));
                %disp(outputweights);
                out=outputweights*out1;
                %disp(out);
                %out=out+bias;
                val=T(i)*nmax;
                tar=zeros(size(out,1),1);
                tar(int16(val),1)=1;
                for j=1:size(out,1)
                    out(j)=1/(1+exp(-out(j))); 
                    delta(j,1)=(out(j)-tar(j))*out(j)*(1-out(j));         
                end
                %disp(out);
                %disp(delta);
                %Updating output weights 10*4
                for k=1:size(outputweights,1)           
                    for l=1:size(outputweights,2)
                        outputweights(k,l)=outputweights(k,l)-(alpha*out1(l)*delta(k, 1));               
                        %disp(A(i,l));
                    end           
                end
                % Update hidden layer weights
                %disp(size(out1,1));
                %disp(size(outputweights));
                %disp(outputweights);
                for jj=1:size(out1,1)
                    sumterm=0;
                    for kk=1:size(outputweights,1)
                        ind=delta(kk)*outputweights(kk,jj);
                        sumterm=sumterm+ind;
                    end
                    delta2(jj,1)=sumterm*(out1(jj)*(1-out1(jj)));
                end
                %disp(delta2);
                %disp(size(delta2));
                for ku=1:size(weightfirst,1)           
                    for lu=1:size(weightfirst,2)
                        weightfirst(ku,lu)=weightfirst(ku,lu)-(alpha*A(i,lu)*delta2(ku));               
                        %disp(A(i,l));
                    end           
                end
                %disp(weightfirst);
                %disp(outputweights);
            end
            alpha=alpha*0.98;
        end
        %disp(outputweights);
        %disp(weightfirst);
        %%---------------->TESTING
        caccuracy=0;
        B=load(test_file);
        ttmax=max(B);
        tmax=max(ttmax);
        B=B/tmax;
        Tt=double(B(:,end));
        Tt=Tt+0.01;
        B=double(B(:,1:end-1));
        biast=zeros(size(B,1),1);
        biast=biast+1;
        B=horzcat(B,biast);
        for i=1:size(B,1)
        %for i=1:numbers
            testout1=weightfirst*transpose(B(i,1:end)); 
            for j=1:size(testout1,1)
                    testout1(j)=1/(1+exp(-testout1(j))); 
            end
            testout1=vertcat(testout1,1);
            testout=outputweights*testout1;
            for j=1:size(testout,1)
                testout(j)=1/(1+exp(-testout(j)));                            
            end
            [m,k]=max(testout);
            vec= find(testout == m);
            tiesize=size(vec,1);
            tar=Tt(i)*tmax;
            if (k)==int16(tar)
                accuracy=1;
            %disp(accuracy);
            elseif tiesize>1
                accuracy=1/tiesize;
            else
                accuracy=0;
            end
            caccuracy=caccuracy+accuracy;
            %fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',i,k-1,int16(tar)-1,accuracy);
        end
        fprintf('classification accuracy=%6.4f\n', caccuracy/size(B,1));
    %%-----------------------------hidden layers>1
    else
        %weightfirst=zeros(unitslayer,size(A,2));
        %weightfirst=weightfirst+0.5;
        weightfirst = -0.05 + (0.05- (-0.05)).*rand(unitslayer,size(A,2));
        biases=zeros(unitslayer,1);
        biases=biases+1;
        weightfirst=horzcat(weightfirst,biases);
        biasi=biasi+1;
        A=horzcat(A,biasi);
        for i=1:size(A,1)
            outhfirst=weightfirst*transpose(A(i,1:end));
            %outhfirst=outhfirst+bias;
            for j=1:size(outhfirst,1)
              outhfirst(j)=1/(1+exp(-outhfirst(j)));
            end            
            %disp(outhfirst)
            outhfirst=vertcat(outhfirst,1);
            outh(:,:,1)=outhfirst;
            %disp('layer1');
            disp(outh(:,:,1));

        end                
        for i = 2:noofhiddenlayers
            disp(i);
            biash=zeros(unitslayer,1);
            biash=biash+1;
            hw=zeros(unitslayer,unitslayer);
            hw=hw+0.5;
            hw=horzcat(hw,biash);
            hweight(:,:,i)=hw;
            outhh=hweight(:,:,i)*outh(:,:,i-1);
            %outhh=outhh+bias;
            for j=1:size(outhh,1)
              outh(j)=1/(1+exp(-outhh(j)));
            end
            %outh=vertcat(outh,1);
            outh(:,:,i)=outh(j);
            %disp(outh(j));
        %disp(noofhiddenlayers)
        %disp(size(outh));
        %disp(outh(:,:,noofhiddenlayers));
        end
        disp(size(outputweights));
        %disp(size(outh(:,:,nohiddenlayers)));
        out=outputweights*outh(:,:,noofhiddenlayers);
        %out=out+1;
        for j=1:size(out,1)
            out(j)=1/(1+exp(-out(j)));
        end
        %disp('layer3');
        disp(out);
    end
 %-----------------------only 2 layers
else
    %outputweights=zeros(noofunitsoutputlayer,size(A,2));
    %outputweights=outputweights+0.5;
    %disp(size(A,2));
    outputweights=-0.05+(0.05+0.05)*rand(noofunitsoutputlayer,size(A,2));
    biass=zeros(noofunitsoutputlayer,1);
    biass=biass+1;
    outputweights=horzcat(outputweights,biass);
    biasi=zeros(size(A,1),1);
    biasi=biasi+1;
    A=horzcat(A,biasi);
    %outputweights=-0.05+(0.05+0.05)*rand(noofunitsoutputlayer,size(A,2));
    %disp(outputweights);
    m=0;
    %------------TRAINING
    while m<rounds
        m=m+1;
        for i=1:size(A,1)        
            %disp(A(i,1:end));
            %out=outputweights*transpose(A(i,1:end));
            %out=out+bias; 
            %disp(out);
            %disp(outputweights);
            out=outputweights*transpose(A(i,1:end)); 
            %disp(out);
            val=T(i)*nmax;
            %disp(int16(val));
            tar=zeros(size(out,1),1);
            tar(int16(val),1)=1;
            for j=1:size(out,1)
                out(j)=1/(1+exp(-out(j))); 
                delta(j,1)=(out(j)-tar(j))*out(j)*(1-out(j));         
            end                    
            %disp(out);
            %disp(delta);
            % Update weights 
            for k=1:size(outputweights,1)           
                for l=1:size(outputweights,2)
                    outputweights(k,l)=outputweights(k,l)-(alpha*A(i,l)*delta(k));               
              %disp(A(i,l));
                end           
            end
       
            %disp(outputweights);
        end
        alpha=alpha*0.98;
    end
    %disp(outputweights);
    %------>Testing
    caccuracy=0;
    B=load(test_file);
    ttmax=max(B);
    tmax=max(ttmax);
    B=B/tmax;
    Tt=double(B(:,end));
    Tt=Tt+0.01;
    B=double(B(:,1:end-1));
    biast=zeros(size(B,1),1);
    biast=biast+1;
    B=horzcat(B,biast);
    for i=1:size(B,1)
        testout=outputweights*transpose(B(i,1:end)); 
        %disp(testout);
        for j=1:size(testout,1)
           testout(j)=1/(1+exp(-testout(j)));
        end
        [m,k]=max(testout);
        vec= find(testout == m);
        tiesize=size(vec,1);
        tar=Tt(i)*tmax;
        if (k)==int16(tar)
            accuracy=1;
            %disp(accuracy);
        elseif tiesize>1
            accuracy=1/tiesize;
        else
            accuracy=0;
            %disp(testout);
        end
        caccuracy=caccuracy+accuracy;
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n',i,k-1,int16(tar)-1,accuracy);
    end
    fprintf('classification accuracy=%6.4f\n', caccuracy/size(B,1));
end
end