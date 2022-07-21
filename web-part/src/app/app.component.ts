import { Component } from '@angular/core';

interface tui {
  name: string;
  count: number;
  f1score: number;
}

interface f1score_tui {
  id: string;
  f1score: number;
}

interface all_f1scores {
  accuracy_model: 0.495;
  model_config: {
    class: string;
    features: string[];
    max_data_per_class: number;
  };
  tuis: f1score_tui[];
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {

  search = '';
  source = 'Unknown source';
  total = 0;
  loadingTTL = false;
  tuis: tui[] = []
  files: any[] = [];
  errorParsingFile = false;

  constructor() { }

  getAllF1Scores() {
    const PATH = 'assets/tui_f1_scores.json';
    return new Promise((resolve) => {
      fetch(PATH)
      .then(response => response.json())
      .then(data => {
        const obj = data as all_f1scores;
        this.tuis.forEach((tui: tui) => {
          const index = obj.tuis.findIndex((tui_obj: f1score_tui) => {
            return tui_obj.id === tui.name;
          });
          if (index !== -1) {
            tui.f1score = obj.tuis[index].f1score;
          }
        });
        resolve(true);
      });
    });
  }

  sortTuis() {
    this.tuis.sort((a, b) => {
      return b.count - a.count;
    });
  }

  countTotal() {
    this.total = this.tuis.reduce((acc, cur) => {
      return acc + cur.count;
    }, 0);
  }

  getColor(f1Score: number) {
    if(f1Score >= 0.7) {
      return 'success';
    } else if (f1Score >= 0.4) {
      return 'warning';
    } else {
      return 'danger';
    }
  }

  getFrequency(count: number) {
    return Math.round(count / this.total * 100);
  }

  parseContent(content: any) {
    try {
      if (content) {
        const prefix1 = "@prefix bpm: <http://bioportal.bioontology.org/ontologies/umls/> ."
        const prefix2 = "@prefix sty: <http://purl.bioontology.org/ontology/STY/> ."
        if (!content.includes(prefix1) || !content.includes(prefix2)) {
          throw new Error('Invalid file format');
        }

        const lines = content.split('\n');
        this.source = this.files[0].name;
        lines.forEach((line: string) => {
          const parts = line.split(' ');
          if (parts.length === 4 && parts[0][0] != '@') {
            const name = parts[2].split(':')[1]
            // Check if tuis already contains the name
            const index = this.tuis.findIndex((tui: tui) => {
              return tui.name === name;
            });
            if (index === -1) {
              this.tuis.push({
                name: name,
                count: 1,
                f1score: 0,
              });
            } else {
              this.tuis[index].count++;
            }
            this.total++;
          }
        }
        );
        this.sortTuis();
        this.countTotal();
        this.getAllF1Scores().then(() => {
          this.loadingTTL = false;
        });
      }
    } catch (error) {
      this.loadingTTL = false;
      this.errorParsingFile = true;
      console.log(error);
    }
  }

  /**
   * on file drop handler
   */
  onFileDropped($event: any) {
    this.files = [];
    this.prepareFilesList($event);
  }

  /**
   * handle file from browsing
   */
  fileBrowseHandler(event: any) {
    this.files = [];
    const element = event.currentTarget as HTMLInputElement;
    let fileList: FileList | null = element.files;
    if (fileList) {
      console.log("FileUpload -> files", fileList);
      this.prepareFilesList(fileList);
    }
  }

  /**
   * Delete file from files list
   * @param index (File index)
   */
  deleteFile(index: number) {
    this.files = [];
  }

  /**
   * Convert Files list to normal array list
   * @param files (Files List)
   */
  prepareFilesList(files: any) {
    for (const item of files) {
      this.files.push(item);
    }
    this.errorParsingFile = false;
    this.getFileContent(this.files[0])
    .then((content: any) => {
      this.parseContent(content);
    });
  }

  getFileContent(file: any) {
    this.loadingTTL = true;
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        resolve(e.target.result);
      }
      reader.readAsText(file);
    });
  }

  /**
   * format bytes
   * @param bytes (File size in bytes)
   * @param decimals (Decimals point)
   */
  formatBytes(bytes: any, decimals?: any) {
    if (bytes === 0) {
      return '0 Bytes';
    }
    const k = 1024;
    const dm = decimals <= 0 ? 0 : decimals || 2;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  resetClick($event: any) {
    $event.target.value = null
  }
}
