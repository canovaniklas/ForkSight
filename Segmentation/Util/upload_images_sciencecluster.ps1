$base = "C:\Users\juhe9\repos\MasterThesis\ForkSight\Segmentation\Data\RawData"
$remoteimages = "sciencecluster:/home/jhehli/data/raw_data/images_4096"
$remotemasks = "sciencecluster:/home/jhehli/data/raw_data/masks_4096"

$images = Join-Path $base "images_4096"
$masks  = Join-Path $base "masks_4096"

Write-Host "Uploading images..."
scp (Join-Path $images "*") $remoteimages

Write-Host "Uploading masks..."
scp (Join-Path $masks "*") $remotemasks

Write-Host "Done!"
