param(
    [string]$ScriptPath = "$(Join-Path $PSScriptRoot 'monte_carlo_option_pricing.py')"
)

python $ScriptPath
