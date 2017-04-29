function [c]=cost(x1,x2,y1,y2)
first=(x1-x2)^2;
sec=(y1-y2)^2;
c=sqrt(first+sec);
end