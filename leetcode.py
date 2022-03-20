import typing
from typing import List
import copy
import numpy as np

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        # dp = [[0]*len(grid)]*len(grid[0])
        dp = copy.deepcopy(grid)

        dp[0][0] = grid[0][0]

        for j in range(1, len(grid[0])):
            dp[0][j]=dp[0][j-1] + grid[0][j]
        
        for i in range(1, len(grid)):
            dp[i][0]=dp[i-1][0] + grid[i][0]

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])+grid[i][j]
        print(dp)
        return dp[len(grid)-1][len(grid[0])-1]


    # 
    def lengthOfLongestSubstring(self, s: str) -> int:
        hash1 = {}
        dp = [0]*len(s)
        if(len(s)==0):
            return 0

        max1 = 1
        for idx, val in enumerate(s):
            if idx==0:
                hash1[val]=0
                dp[idx] = 1
                continue

            # j-i > dp[j-1]的意思：当前字符与上一个重复字符之间的距离如果超过了前一位字符的dp值，则表明，当前字符与重复的字符的距离比前一字符非重复区间还要长，此时满足当前字符与前一字符的dp转移关系

            # j-i <= dp[j-1]的意思：当前字符与上一个重复字符之间的距离未超过前一位字符的dp值，则表明，重复的字符出现在了前一位字符的非重复区间了
            if (idx - hash1.get(val, -1))>dp[idx-1]:
                hash1[val] = idx
                dp[idx] = dp[idx-1]+1
            else:
                dp[idx] = idx - hash1[val]
                hash1[val] = idx
            max1 = max(max1, dp[idx])
        return max1

    def minDistance(self, word1: str, word2: str) -> int:
        dp = np.zeros((len(word1)+1, len(word2)+1))

        dp[0] = np.arange(len(word2)+1)
        dp[:,0] = np.arange(len(word1)+1)
        
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                dp[i][j] = min(
                    dp[i-1][j]+1,
                    dp[i][j-1]+1,
                    dp[i-1][j-1] + (1 if word1[i-1]!=word2[j-1] else 0)
                )
        return int(dp[len(word1)][len(word2)])

    def singleNumbers(self, nums: List[int]) -> List[int]:
        m, x, y, p = 0, 0, 0, 1

        for i in nums:
            m ^= i
        while m&p == 0:
            # if m&p == 1:
            #     break
            p<<=1
        for i in nums:
            if i&p==0:
                y^=i
            else:
                x^=i
        return [x,y]

    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return len(nums)
        dp = [1] * len(nums)
        result = 0
        for i in range(1, len(nums)):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            result = max(result, dp[i]) #取长的子序列
        return result

    # --------

    def is_palin(self, i:int, j:int, s:str) -> str:
        while i>=0 and j<len(s):
            if s[i]!=s[j]:
                return s[i+1: j]
            i-=1
            j+=1
        return s[i+1: j]

    def longestPalindrome(self, s: str) -> str:

        res = ""

        for i in range(0, len(s)):
            left = self.is_palin(i, i, s)
            right = self.is_palin(i, i+1, s)
            if len(left)>len(res):
                res = left
            if len(right)>len(res):
                res = right
        return res

# 
    def removeDuplicates(self, nums: List[int]) -> int:
        # 最后的索引
        index=0
        # 重复的值
        left=1
        # 待检查的值
        right=1

        while right<len(nums):
            if nums[right]!=nums[index]:
                nums[left], nums[right] = nums[right], nums[left]
                index = left
                left += 1
            right+=1
        print(nums)
        return index+1

# 

    def reverseWords(self, s: str) -> str:
        def rev(sss):
            sss = list(sss)
            left,right=0,len(sss)-1
            while left<right:
                sss[left],sss[right]=sss[right],sss[left]
                left+=1
                right-=1
            return ''.join(sss)
        start=0
        candi=''
        for i,val in enumerate(s):
            if val == ' ':
                candi+=rev(s[start:i])
                start=i 
                candi+=' '
            i+=1
        return candi
# 
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        x = [1]*len(nums)
        y = [1]*len(nums)
        pre=nums[0]
        for i in range(1, len(nums)):
            x[i] = pre*x[i-1]
            pre=nums[i]

        pre=nums[-1]
        for i in range(len(nums)-2,-1,-1):
            y[i] = pre*y[i+1]
            pre=nums[i]
        
        # mul=[1]*len(nums)
        # for i in range(len(nums)):
        #     mul[i]=x[i]*y[i]
        # return mul
        pre=nums[-1]
        for i in range(len(nums)-2,-1,-1):
            x[i] = pre*x[i+1]
            pre=nums[i]

        return x

# 

    def generateMatrix(self, n: int) -> List[List[int]]:
        l, r, t, b = 0, n - 1, 0, n - 1
        mat = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1): # left to right
                mat[t][i] = num
                num += 1
            t += 1
            for i in range(t, b + 1): # top to bottom
                mat[i][r] = num
                num += 1
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                mat[b][i] = num
                num += 1
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                mat[i][l] = num
                num += 1
            l += 1
        return mat

# 
    # 合并两个有序链表
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        if n==0:
            return
        if m==0:
            for i,val in enumerate(nums2):
                nums1[i] = val
            return
        first = m-1
        second = n-1
        index = m+n-1

        while first>=0 and second>=0:
            if nums1[first]>nums2[second]:
                nums1[index] = nums1[first]
                first-=1
                index-=1
            else:
                nums1[index] = nums2[second]
                second-=1
                index-=1
        while first>=0:
            nums1[index] = nums1[first]
            index-=1
            first-=1
        while second >=0:
            nums1[index] = nums2[second]
            index-=1
            second-=1
        return



    # 两个链表求和
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1==None:
            return l2
        if l2==None:
            return l1
        
        flag = 0
        dummy = ListNode()
        dum = dummy

        while l1 and l2:
            val = l1.val+l2.val+flag
            flag = val//10
            val = val%10
            dum.next = ListNode(val)
            dum = dum.next
            l1 = l1.next
            l2 = l2.next
        
        while l1:
            val = l1.val+flag
            flag = val//10
            val = val%10

            dum.next = ListNode(val)
            dum = dum.next
            l1 = l1.next
        while l2:
            val = l2.val+flag
            flag = val//10
            val = val%10

            dum.next = ListNode(val)
            dum = dum.next
            l2 = l2.next            
        
        if flag!=0:
            dum.next = ListNode(flag)
            dum = dum.next
    
        return dummy.next

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result = ListNode(0)
        re = result
        carry = 0
        # l1,l2,进位 有一个存在则需要继续链表
        while l1 or l2 or carry > 0:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            re.next = ListNode((x + y + carry) % 10)
            re = re.next
            # 进位
            carry = (x + y + carry) // 10
            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next
        return result.next

    def merge_k_lists(self, lists:List[ListNode]) -> ListNode:
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next

    def maxSubArray(self, nums: List[int]) -> int:
        imax = [0]*len(nums)
        imax[0] = nums[0]
        pos = [0]

        for i in range(1, len(nums)):
            # imax[i] = max(
            #     imax[i-1]+nums[i], nums[i]
            # )

            if imax[i-1]+nums[i] > nums[i]:
                imax[i] = imax[i-1]+nums[i]
                pos.append(i)
            else:
                imax[i] = nums[i]
                pos.clear()
                pos.append(i)

        print(pos[0], pos[-1])
        print(pos)
        return max(imax)


    def maxProduct(self, nums: List[int]) -> int:
        if len(nums)==0:
            return 
        if len(nums)==1:
            return nums[0]
        imax = nums[0]
        imin = nums[0]


        final_max = nums[0]

        for i in range(1, len(nums)):
            bkmin, bkmax = imin, imax
            imin = min(bkmin*nums[i], nums[i], bkmax*nums[i])
            imax = max(bkmax*nums[i], nums[i], bkmin*nums[i])

            final_max = max(imax, final_max)

        return final_max

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA:
            return None
        if not headB:
            return None
        
        
        p1, p2 = headA, headB

        while p1 != p2:
            p1=p1.next if p1 else headB
            p2=p2.next if p2 else headA
        
        return p1
        
    def minDistance2(self, word1: str, word2: str) -> int:
        dp = [[0]*(len(word1)+1)]*(len(word2)+1)
        # dp = np.zeros((len(word1)+1, len(word2)+1))

        # dp[0] = np.arange(len(word2)+1)
        dp[0] = list(range(0, len(word2)+1))

        dp[:][0] = list(range(0, len(word1)+1))
        # dp[:,0] = np.arange(len(word1)+1)

        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                dp[i][j] = min(
                    dp[i-1][j-1]+(1 if word1[i-1]!=word2[j-1] else 0),
                    dp[i-1][j]+1,
                    dp[i][j-1]+1
                )
        return dp[len(word1)][len(word2)]

    def sortList(self, head: ListNode) -> ListNode:
        # https://leetcode-cn.com/problems/7WHec2/solution/ii-077-lian-biao-pai-xu-python-gui-bing-zedml/
        def merge(h):
            if not h or not h.next: return h

            p, q = h, h
            while q and q.next and q.next.next:
                p = p.next
                q = q.next.next
            temp = p.next
            p.next = None
            left = merge(h)

    def findMin(self, nums: List[int]) -> int:

        left=0
        right=len(nums)-1
        ans=0

        while left<right:
            mid=left+(right-left)//2

            # if nums[mid]<nums[mid-1] and nums[mid]<nums[mid+1]:
            #     ans = mid
            #     return nums[mid]
            # elif nums[mid]<nums[mid-1] and nums[mid]>nums[mid+1]:
            #     right=mid-1
            # else:
            #     left=mid+1

            if nums[mid]>nums[right]:
                left=mid+1
            else:
                right=mid
        
        return nums[left]
                    
    def search(self, nums: List[int], target: int) -> int:
        n=len(nums)

        def search_mid(nums:List[int]):
            left, right = 0, len(nums)-1

            while left<right:
                mid=(left+right)//2
                
                if nums[mid]>nums[right]:
                    left=mid+1
                else:
                    right=mid
            return left
        
        mid=search_mid(nums)

        i, j = 0, 0

        if nums[0] <= target <= nums[mid-1]:
            i, j = 0, mid
        elif nums[mid] <= target <= nums[n-1]:
            i, j = mid, n-1
        else:
            return -1
        
        while i<=j :
            mid = (i+j)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                j=mid-1
            else:
                i=mid+1

        return -1


    def findKthLargest(self, nums: List[int], k: int) -> int:
        import heapq
        heap = []
        for i in nums:
            if len(heap)<k:
                heapq.heappush(heap, i)
            else:
                if i > heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, i)
        return heap[0]


    # # 构建堆
    # def myheapify(nums, i, k):
    #     min_index = i
    #     while True:
    #         if i*2+1 < k and nums[i*2+1]<nums[min_index]:
    #             min_index=i*2+1
    #         if i*2+2 < k and nums[i*2+2]<nums[min_index]:
    #             min_index = i*2+2

    #         if min_index==i:
    #             break
    #         nums[i], nums[min_index] = nums[min_index], nums[i]
    #         i = min_index

    # def build_heap(nums, k):
    #     for i in range(k//2-1, -1, -1):
    #         myheapify(nums, i, k)

    def findMin2(self, nums: List[int]) -> int:
        n=len(nums)
        left, right=0, n-1

        while left<right:
            mid=(left+right)//2

            if nums[mid]>nums[right]:
                left=mid+1
            else:
                right=mid
        
        return nums[left]

    def search_again(self, nums: List[int], target: int) -> int:

        def search_mid(nums):
            n=len(nums)
            left, right=0, n-1

            while left<right:
                mid=(left+right)//2

                if nums[mid]>nums[right]:
                    left=mid+1
                else:
                    right=mid
            return left
        
        # main

        n=len(nums)

        min_index = search_mid(nums)
        i, j = 0, n-1

        if min_index==0:
            i, j =0, n-1
        elif nums[0]<=target<=nums[min_index-1]:
            i, j = 0, min_index-1
        elif nums[min_index]<=target<=nums[n-1]:
            i, j = min_index, n-1
        else:
            return -1
        
        while i<=j:
            mid=(i+j)//2
            
            if target==nums[mid]:
                return mid
            elif target>nums[mid]:
                i=mid+1
            else:
                j=mid-1
        
        if nums[mid]==target:
            return mid
        else:
            return -1


    def findTarget(self, root: TreeNode, k: int) -> bool:
        stack = []

        def inner(root):
            if root:
                inner(root.left)
                stack.append(root.val)
                inner(root.right)
        
        inner(root)
    
    # 二叉树最近公共祖先
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        if not root or root.val==p.val or root.val==q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root
        elif left:
            return left
        else:
            return right

    # 二叉搜索树最近公共祖先
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        if root.val>p.val and root.val>q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

    def generateParenthesis(self, n: int) -> List[str]:
        self.ans=[]

        def dfs(left, right, sss):
            if left==0 and right==0:
                self.ans.append(sss)
                return
            # print(left, right)
            if left>0:
                dfs(left-1, right, sss+'(')
            if right>left:
                dfs(left, right-1, sss+')')
            return

        dfs(n,n,"")
        return self.ans
        

    def subsets(self, nums: List[int]) -> List[List[int]]:


        self.ans = []
        self.nums = nums

        def dfs( n1 ):
            if len(n1):
                self.ans.append(n1)
            else:
                return 
            dfs(n1[:-1])
            return
        
        # dfs(nums)

        for i in nums:
            bk = copy.deepcopy(nums)
            bk.remove(i)
            dfs(bk)
        self.ans.append([])

        return self.ans
# 

    def subsets1(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []

        def dfs(start, path):
            ans.append(path[:])
            for i in range(start, n):
                path.append(nums[i])
                dfs(i + 1, path)
                path.pop()
            return

        dfs(0, [])
        return ans


    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.ans=[]

        def dfs(start, n, path):

            self.ans.append(path)
            
            for i in range(start, n):
                dfs(i+1, n, path+[nums[i]])
            return

        dfs(0, len(nums), [])

        return self.ans


    # 输入：nums = [1,2,3]
    # 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

    def permute(self, nums: List[int]) -> List[List[int]]:
        self.ans=[]

        def dfs(start, n, path):
            if len(path) == n:
                self.ans.append(path)
                return
            
            for i in range(0, n):
                if nums[i] in path:
                    continue
                dfs(i+1, n, path+[nums[i]])
            
            return
        
        dfs(0, len(nums), [])

        return self.ans

        
# grid=[
#     [1,3,1],
#     [1,5,1],
#     [4,2,1]
# ]

# print(Solution().maxValue(grid))
# print(Solution().lengthOfLongestSubstring("abba"))
# print(Solution().minDistance("horse", 'ros'))
# print(Solution().singleNumbers([1,2,5,2]))

# print(Solution().lengthOfLIS([1,7,5,9,4,8]))

# print(Solution().longestPalindrome("babad"))

# print(Solution().removeDuplicates([0,1,2,3]))
# print(Solution().reverseWords("Let's take LeetCode contest"))

# print(Solution().productExceptSelf([1,2,3,4]))


# print(Solution().generateMatrix(5))

# print(Solution().merge([4,5,6,0,0,0], 3, [1,2,3], 3))

# print(Solution().maxProduct([0,100,-1,0,50,50]))


# print(Solution().maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))


# print(Solution().minDistance2("horse", 'ros'))


# print(Solution().findMin(
#     [4,5,6,7,0,1,2]
# ))


# print(Solution().search(
#     # [4,5,6,7,0,1,2], 6
#     [1,3], 3
# ))


# print(Solution().findKthLargest(
#     # [4,5,6,7,0,1,2], 6
#     [3,2,1,5,6,4], 2
# ))



# print(Solution().findMin2(
#     [4,5,6,7,0,1,2]
# ))



# print(Solution().search_again(
#     # [4,5,6,7,0,1,2], 0
#     [1,3], 3
# ))

# print(Solution().generateParenthesis(
#     3
#     # [4,5,6,7,0,1,2], 0
# ))


# print(Solution().subsets(
#     [1,2,3]
# ))

print(Solution().permute(
    [1,2,3]
))

