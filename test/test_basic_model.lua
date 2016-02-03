
package.path = '../?.lua;' .. package.path
require 'misc.basic_modules'

print('Testing nn.Scale ... ')
mlp1 = nn.Scale(4)
a = torch.ones(3, 4)

local precision = 1e-5
local jac = nn.Jacobian
local err = jac.testJacobian(mlp1, a)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end


print('\nTesting nn.FeatExpander ... ')
mlp2 = nn.FeatExpander(2)
a = torch.randn(3, 4)

local precision = 1e-5
local jac = nn.Jacobian
local err = jac.testJacobian(mlp2, a)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end

