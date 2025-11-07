<#
.SYNOPSIS
  Push all local changes to the configured Git remote (origin) and branch.

.DESCRIPTION
  Convenience script to stage, commit (when needed), pull/rebase and push changes to GitHub.
  Supports optional token-based authentication via the GITHUB_TOKEN environment variable.

.USAGE
  PowerShell (from repo root):
    .\tools\git_push_all.ps1 -CommitMessage "my changes"

  Common options:
    -CommitMessage <string>   Commit message to use (default: timestamped message).
    -Branch <string>          Branch to push to (default: main).
    -ForcePush                Force push (use with caution).
    -UseToken                 If set and $env:GITHUB_TOKEN exists, inject token into remote URL for push.

.NOTES
  - This script expects to be run from anywhere inside the git repository. It will resolve the
    repository root automatically.
  - Do NOT store GitHub tokens in files. Use environment variables and your OS secret store.
#>

param(
    [string]$CommitMessage = "",
    [string]$Branch = "main",
    [switch]$ForcePush,
    [switch]$UseToken
)

function AbortIfNoGit() {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "git is not installed or not in PATH. Install Git before running this script."
        exit 2
    }
}

try {
    AbortIfNoGit

    # Resolve repo root
    $repoRoot = (git rev-parse --show-toplevel) 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "This directory is not a git repository. Run the script from inside a git repo."
        exit 3
    }
    Set-Location -Path $repoRoot

    if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
        $CommitMessage = "chore: sync workspace $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')"
    }

    Write-Host "Repository root: $repoRoot"
    Write-Host "Branch: $Branch"

    # Stage everything
    Write-Host "Staging all changes..."
    git add -A

    # Only commit if there are staged changes
    $porcelain = git status --porcelain
    if (-not [string]::IsNullOrWhiteSpace($porcelain)) {
        Write-Host "Committing changes..."
        git commit -m $CommitMessage
    } else {
        Write-Host "No changes to commit."
    }

    # Fetch & rebase latest from origin
    Write-Host "Fetching and rebasing from origin/$Branch..."
    git fetch origin $Branch
    git pull --rebase origin $Branch

    # Determine remote URL
    $originUrl = git remote get-url origin

    if ($UseToken -and $env:GITHUB_TOKEN) {
        if ($originUrl -like 'https://*') {
            $token = $env:GITHUB_TOKEN
            $maskedPushUrl = $originUrl -replace '^https://', "https://$token@"
            Write-Host "Temporarily setting remote to token-authenticated URL for push..."
            git remote set-url origin $maskedPushUrl
            try {
                if ($ForcePush) { git push origin $Branch --force } else { git push origin $Branch }
            } finally {
                Write-Host "Restoring original origin URL..."
                git remote set-url origin $originUrl
            }
        } else {
            Write-Warning "Origin remote is not an HTTPS URL. Skipping token injection and pushing with existing credentials."
            if ($ForcePush) { git push origin $Branch --force } else { git push origin $Branch }
        }
    } else {
        Write-Host "Pushing to origin/$Branch using configured credentials..."
        if ($ForcePush) { git push origin $Branch --force } else { git push origin $Branch }
    }

    Write-Host "Push complete."
    exit 0
} catch {
    Write-Error "An error occurred: $_"
    exit 4
}
