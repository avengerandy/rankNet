let itemRankedATag = document.getElementsByClassName('work_thumb_inner');
let itemIds = [];
var itemIdPattern = new RegExp('\.([A-Z]+[0-9]+)\.html');
for (let i = 0; i < itemRankedATag.length; i++) {
    let hrefUrl = itemRankedATag[i].getAttribute("href");
    let match = itemIdPattern.exec(hrefUrl) 
    itemIds.push(match[1]);
}
console.log(JSON.stringify(itemIds));
