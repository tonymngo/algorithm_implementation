import math

def merge(a,left,right,split):
	#merge 2 sorted arrays
	L = a[left:split+1]
	R = a[split+1:right+1]
	ans = []
	i,j = 0,0
	while (i < len(L)) and (j < len(R)):
		if L[i] <= R[j]:
			ans.append(L[i])
			i+=1
		else:
			ans.append(R[j])
			j+=1
	if len(L[i:])>0:
		ans.extend(L[i:])
	else:
		ans.extend(R[j:])
	a[left:right+1] = ans
	return a[left:right+1]

def merge_sort(a,left,right):
	#recursively sort smaller arrays and then merge them together
	if left == right:
		return a[left]
	else:
		split = left + int(math.floor((float(right-left))/2))
		merge_sort(a,left,split)
		merge_sort(a,split+1,right)
		return merge(a,left,right,split)

a = [23,324,36,3,234,2,345,67,56,734,5,345,345,3,4,123,34,435,36,45,7665,75,6653,214,2334,234,2]
print merge_sort(a,0,len(a)-1)

# a = input("Enter a list: ")
# left = input("Sorted from: ")
# right = input("To: ")
