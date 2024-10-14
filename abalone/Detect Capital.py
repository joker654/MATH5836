class Solution:
    def detectCapitalUse(self,word: str) -> bool:
        # All characters are uppercase
        if word.upper() == word:
            return True

        # All characters are lowercase
        if word.lower() == word:
            return True

        if word.capitalize() == word:
            return True

        return False

sol = Solution()
print(sol.detectCapitalUse("USA"))
print(sol.detectCapitalUse("FlaG"))
print(sol.detectCapitalUse("ASAP"))
print(sol.detectCapitalUse("LEeD"))
print(sol.detectCapitalUse("cccp"))
