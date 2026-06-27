import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Visit } from '../models/visit.model';

@Injectable({
  providedIn: 'root',
})
export class VisitService {
  private baseUrl = 'http://127.0.0.1:5000/api/visits';

  constructor(private http: HttpClient) {}

  // POST: /api/visits (Započinjanje novog pregleda)
  createNewExam(patientId: number): Observable<Visit> {
    return this.http.post<Visit>(this.baseUrl, { patient_id: patientId });
  }

  // GET: /api/visits/patient/:id (Svi pregledi jednog pacijenta)
  getExamsByPatient(patientId: number): Observable<Visit[]> {
    return this.http.get<Visit[]>(`${this.baseUrl}/patient/${patientId}`);
  }

  // GET: /api/visits/:id
  getExam(id: number): Observable<Visit> {
    return this.http.get<Visit>(`${this.baseUrl}/${id}`);
  }

  // POST: /api/visits/:id/upload-images (Multipart/form-data za slanje fajla)
  uploadFundusImage(visitId: number, imageFile: File): Observable<any> {
    const formData = new FormData();
    // Ključ mora biti 'image_OD' jer je tako definisano u Insomniji
    formData.append('image_OD', imageFile, imageFile.name);

    return this.http.post<any>(`${this.baseUrl}/${visitId}/upload-images`, formData);
  }
}
