function[tree,threshold,gain]=dtl(example,pruning_thr,option,attributes,nmax,tree,threshold,gain,index)
example=double(example);
T=double(example(:,end));
u=unique(T);
%if size(example,1)<= pruning_thr
if size(example,1)<= pruning_thr
    %disp('inside if \n');
    %disp(index);
    distribution=zeros(nmax,1);
    T = example(1:end,end); 
    for j=1:size(distribution,1)
        vec= find(T==j);
        probclass=size(vec,1)/size(T,1);
        distribution(j,1)=probclass;
    end
    %
    %dist = histc(T(:),unique(T));
    %dist = dist / size(T,1);
    [m,i]=max(distribution);
    tree(index)=i;
    threshold(index)=-1;
    gain(index)=-1;
    %disp(tree);
elseif size(u,1)==1
    %disp('inside elseif \n');
    tree(index)=u;
    threshold(index)=-1;
    gain(index)=-1;
    %disp(tree);
    %disp(index);
    %disp(u);
else
    [best_attri,best_thresh,best_gain]= chooseAttribute(example, attributes,option);
    %disp(best_attri);
    %fprintf('best_attri=%f, best_thresh=%f \n',best_attri, best_thresh);
    %disp(best_attri);
    %disp(best_thresh);
    %rightexample=[];
    %leftexample=[];
    cr=1;
    cl=1;
    tree(index)=best_attri;
    threshold(index)=best_thresh;
    gain(index)=best_gain;
    %leftexample=[];
    %rightexample=[];
    for i=1:size(example,1)
       if example(i,best_attri) >= best_thresh
            rightexample(cr,:)=example(i,1:end);            
            cr = cr + 1;
       else            
            leftexample(cl,:) = example(i,1:end);
            cl = cl + 1;
        end 
    end
    %disp(size(leftexample));
    %disp(size(rightexample));
    %disp(tree);
    if exist('leftexample')
       %disp('true');
       [tree,threshold,gain]= dtl(leftexample,pruning_thr,option,attributes,nmax,tree,threshold,gain,2*index);    
       
    else
        %disp('false');
        tree(2*index)=1;
        threshold(2*index)=-1;
        gain(2*index)=-1;
    end
    if exist('rightexample')
        [tree,threshold,gain] = dtl(rightexample,pruning_thr,option,attributes,nmax,tree,threshold,gain,(2*index)+1);
    else
        tree((2*index)+1)=1;
        threshold((2*index)+1)=-1;
        gain((2*index)+1)=-1;
    end
    %[tree,threshold,gain]= dtl(leftexample,pruning_thr,option,attributes,nmax,tree,threshold,gain,2*index);    
    %[tree,threshold,gain] = dtl(rightexample,pruning_thr,option,attributes,nmax,tree,threshold,gain,(2*index)+1);
    
    %disp(best_thresh);
    %disp(rightexample);
    %disp(leftexample);
end
%disp(gain);
end